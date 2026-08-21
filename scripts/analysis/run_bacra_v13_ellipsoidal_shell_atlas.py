#!/usr/bin/env python3
"""Build the BACRA V13 physically thick ellipsoidal-shell dataset.

The primary stopping rule is intentionally before Student training: the run
must first demonstrate same-chart surface/volume coverage, physical normal
thickness, parent agreement, FK residual, bounds, and near-uniform volume
sampling.  A Student result can consume this dataset but cannot upgrade a
failed shell Gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import norm, qmc
import yaml

from quasi_exp.teacher.canonical import TeacherPolicy, weighted_damped_pinv
from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasPolicy,
    AtlasTaskNode,
    build_canonical_atlas,
    make_predictor_corrector_continuation,
)
from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.ellipsoidal_shell import (
    EllipsoidSpec,
    build_shell_mesh,
    covered_cells,
    radial_cell_volumes,
    sample_shell_cells,
    shell_coverage_report,
)
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.multi_ik_candidates import (
    CandidatePolicy,
    CandidateQuality,
    solve_candidate_bank,
)
from quasi_exp.teacher.region_growth import (
    RegionLabelPolicy,
    continue_from_parent,
    reduce_parent_candidates,
)
from run_trajectory_canonical_teacher_v10 import load_environment, runtime_fingerprint


PROTOCOL_ID = "bacra-v13-ellipsoidal-shell-canonical-atlas"
CLAIM_SCOPE = "simulation_ellipsoidal_shell_known_chart_dataset"
DEFAULT_PYTHON = Path("/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python")
STAGES = (
    ("protocol", "00_protocol"),
    ("shell_search", "01_shell_search"),
    ("surface_atlas", "02_surface_atlas"),
    ("radial_labels", "03_radial_labels"),
    ("dense_dataset", "04_dense_dataset"),
    ("summary", "05_summary"),
)
STAGE_DIR = dict(STAGES)


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    if ".worktrees" in resolved.parts:
        return Path(*resolved.parts[: resolved.parts.index(".worktrees")])
    return resolved


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, preset: str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V13 config must contain a mapping")
    profiles = dict(payload.get("presets", {}))
    if preset not in {"smoke", "pilot", "formal"}:
        raise ValueError("preset must be smoke, pilot, or formal")
    override = profiles.get(preset, {})
    merged = _deep_merge({key: value for key, value in payload.items() if key != "presets"}, override)
    merged["preset"] = preset
    merged["config_path"] = str(config_path)
    return merged


def _source_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _output_root(config: Mapping[str, Any], project_root: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    root = _source_path(project_root, str(config["output_root"]))
    if config["preset"] == "pilot":
        return root.with_name(root.name + "_pilot")
    if config["preset"] == "formal":
        return root.with_name(root.name + "_formal")
    return root


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _gate(path: Path, checks: Mapping[str, bool], *, semantics: str, **evidence: Any) -> dict[str, Any]:
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


def _require_gate(output_root: Path, stage_name: str) -> dict[str, Any]:
    path = output_root / STAGE_DIR[stage_name] / "gate.json"
    if not path.is_file():
        raise FileNotFoundError(f"required V13 gate is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("gate_pass")):
        raise RuntimeError(f"required V13 gate failed: {path}")
    return payload


def _sources(config: Mapping[str, Any], project_root: Path) -> dict[str, Path]:
    values = config["sources"]
    return {
        name: _source_path(project_root, values[name])
        for name in (
            "capability_pool",
            "capability_beta",
            "chart_A_sparse",
            "chart_B_sparse",
            "comparison_dataset",
        )
    }


def stage_protocol(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    stage = output_root / STAGE_DIR["protocol"]
    stage.mkdir(parents=True, exist_ok=False)
    sources = _sources(config, project_root)
    expected = dict(config["sources"]["expected_sha256"])
    actual = {name: sha256_file(path) if path.is_file() else "missing" for name, path in sources.items()}
    robot = _source_path(project_root, config["robot_config"])
    git_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    implementation = {
        "runner": Path(__file__).resolve(),
        "config": Path(config["config_path"]).resolve(),
        "shell_module": SOURCE_ROOT / "src/quasi_exp/teacher/ellipsoidal_shell.py",
        "candidate_module": SOURCE_ROOT / "src/quasi_exp/teacher/multi_ik_candidates.py",
        "atlas_module": SOURCE_ROOT / "src/quasi_exp/teacher/canonical_atlas.py",
        "region_module": SOURCE_ROOT / "src/quasi_exp/teacher/region_growth.py",
    }
    atomic_write_json(stage / "frozen_config.json", {key: value for key, value in config.items() if key != "config_path"})
    atomic_write_json(
        stage / "source_manifest.json",
        {
            "git_sha": git_sha,
            "source_worktree": str(SOURCE_ROOT),
            "source_artifacts": {
                name: {"path": str(path), "sha256": actual[name], "expected_sha256": expected[name]}
                for name, path in sources.items()
            },
            "implementation": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in implementation.items()
            },
        },
    )
    atomic_write_json(
        stage / "runtime.json",
        {**runtime_fingerprint(), "hostname": platform.node(), "cpu_count": os.cpu_count(), "python": sys.executable},
    )
    return _gate(
        stage / "gate.json",
        {
            "source_files_exist": all(path.is_file() for path in sources.values()),
            "source_hashes_match": actual == expected,
            "robot_config_exists": robot.is_file(),
            "source_worktree_clean": git_status.strip() == "",
            "protocol_id_matches": config["protocol_id"] == PROTOCOL_ID,
            "claim_scope_matches": config["claim_scope"] == CLAIM_SCOPE,
            "deployment_claim_disabled": config["deployment_claim_gate_pass"] is False,
        },
        semantics="clean_fixed_point_and_exact_source_bytes",
        git_sha=git_sha,
        source_sha256=actual,
    )


def _capability_frame(path: Path) -> pd.DataFrame:
    columns = [
        *XYZ_COLUMNS,
        *BETA_COLUMNS,
        "minimum_margin_deg",
        "sigma3_m",
        "kappa",
        "capability_tier",
    ]
    frame = pd.read_parquet(path, columns=columns)
    finite = np.isfinite(frame.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS, "minimum_margin_deg", "sigma3_m", "kappa"]]).all(axis=1)
    return frame.loc[finite & frame["capability_tier"].eq("Core-safe")].reset_index(drop=True)


def _proper_pca(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(xyz, axis=0)
    covariance = np.cov((xyz - center).T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    if np.linalg.det(vectors) < 0.0:
        vectors[:, -1] *= -1.0
    local = (xyz - center) @ vectors
    return center, vectors, local, np.sqrt(values[order])


def _support_mask(
    targets: np.ndarray,
    all_tree: cKDTree,
    good_tree: cKDTree,
    *,
    radius_m: float,
    count_min: int,
) -> np.ndarray:
    distances, _indices = all_tree.query(targets, k=int(count_min), distance_upper_bound=radius_m)
    kth = distances if int(count_min) == 1 else distances[:, -1]
    good_distance, _ = good_tree.query(targets, k=1, distance_upper_bound=radius_m)
    return np.isfinite(kth) & np.isfinite(good_distance)


def _connected_area_ratio(mesh: Any, supported: np.ndarray) -> float:
    active_faces = np.flatnonzero(np.all(supported[mesh.faces], axis=1))
    if not len(active_faces):
        return 0.0
    face_by_vertex: list[list[int]] = [[] for _ in mesh.vertices_m]
    for face_id in active_faces:
        for vertex_id in mesh.faces[face_id]:
            face_by_vertex[int(vertex_id)].append(int(face_id))
    remaining = set(int(value) for value in active_faces)
    best = 0.0
    while remaining:
        stack = [remaining.pop()]
        component: list[int] = []
        while stack:
            face_id = stack.pop()
            component.append(face_id)
            neighbors: set[int] = set()
            for vertex_id in mesh.faces[face_id]:
                neighbors.update(face_by_vertex[int(vertex_id)])
            linked = neighbors & remaining
            remaining.difference_update(linked)
            stack.extend(linked)
        best = max(best, float(np.sum(mesh.face_areas_m2[component])))
    return best / mesh.surface_area_m2


def _candidate_specs(
    xyz: np.ndarray,
    *,
    count: int,
    seed: int,
    semiaxis_bounds_m: tuple[float, float],
) -> tuple[EllipsoidSpec, ...]:
    center, pca, local, std = _proper_pca(xyz)
    lower = np.quantile(local, 0.10, axis=0)
    upper = np.quantile(local, 0.90, axis=0)
    specs: list[EllipsoidSpec] = []
    for factors in ((0.65, 0.65, 0.65), (0.80, 0.72, 0.80), (0.95, 0.80, 0.90), (1.05, 0.90, 1.00)):
        axes = np.clip(std * np.asarray(factors), semiaxis_bounds_m[0], semiaxis_bounds_m[1])
        specs.append(EllipsoidSpec(center, np.sort(axes)[::-1], pca))
    remaining = max(0, int(count) - len(specs))
    if remaining:
        engine = qmc.Sobol(d=10, scramble=True, seed=int(seed))
        samples = engine.random_base2(m=int(math.ceil(math.log2(remaining))))[
            :remaining
        ]
        for row in samples:
            local_center = lower + row[:3] * (upper - lower)
            candidate_center = center + local_center @ pca.T
            quaternion = norm.ppf(np.clip(row[3:7], 1.0e-9, 1.0 - 1.0e-9))
            quaternion /= np.linalg.norm(quaternion)
            rotation = Rotation.from_quat(quaternion).as_matrix()
            log_min, log_max = np.log(semiaxis_bounds_m)
            raw_axes = np.exp(log_min + row[7:10] * (log_max - log_min))
            specs.append(EllipsoidSpec(candidate_center, np.sort(raw_axes)[::-1], rotation))
    return tuple(specs)


def stage_shell_search(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require_gate(output_root, "protocol")
    stage = output_root / STAGE_DIR["shell_search"]
    stage.mkdir(parents=True, exist_ok=False)
    pool = _capability_frame(_sources(config, project_root)["capability_pool"])
    xyz = pool.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    good = pool["minimum_margin_deg"].to_numpy(dtype=float) >= float(config["search"]["good_margin_min_deg"])
    all_tree, good_tree = cKDTree(xyz), cKDTree(xyz[good])
    radius_m = float(config["search"]["support_radius_mm"]) / 1000.0
    count_min = int(config["search"]["support_count_min"])
    subdivisions = int(config["mesh"]["subdivisions"])
    half = float(config["search"]["probe_half_thickness_mm"]) / 1000.0
    axis_bounds = tuple(float(value) / 1000.0 for value in config["search"]["semiaxis_bounds_mm"])
    envelope_q = tuple(float(value) for value in config["search"]["envelope_quantiles"])
    envelope_lower, envelope_upper = np.quantile(xyz, envelope_q, axis=0)
    reports: list[dict[str, Any]] = []
    ranked: list[tuple[tuple[float, ...], EllipsoidSpec, Any, tuple[np.ndarray, ...]]] = []
    for candidate_id, spec in enumerate(
        _candidate_specs(
            xyz,
            count=int(config["search"]["candidate_count"]),
            seed=int(config["search"]["seed"]),
            semiaxis_bounds_m=axis_bounds,
        )
    ):
        mesh = build_shell_mesh(spec, subdivisions)
        probes = tuple(mesh.offset_vertices(rho) for rho in (-half, 0.0, half))
        envelope_pass = bool(
            np.all(np.vstack(probes) >= envelope_lower - 1.0e-12)
            and np.all(np.vstack(probes) <= envelope_upper + 1.0e-12)
        )
        masks = tuple(
            _support_mask(value, all_tree, good_tree, radius_m=radius_m, count_min=count_min)
            if envelope_pass else np.zeros(len(mesh.vertices_m), dtype=bool)
            for value in probes
        )
        ratios = tuple(float(np.mean(value)) for value in masks)
        connected = _connected_area_ratio(mesh, masks[1])
        volume = float(np.prod(spec.semiaxes_m))
        key = (ratios[1], min(ratios[0], ratios[2]), connected, volume)
        reports.append(
            {
                "candidate_id": candidate_id,
                "center_x_m": spec.center_m[0], "center_y_m": spec.center_m[1], "center_z_m": spec.center_m[2],
                "semiaxis_a_m": spec.semiaxes_m[0], "semiaxis_b_m": spec.semiaxes_m[1], "semiaxis_c_m": spec.semiaxes_m[2],
                "minus_support_ratio": ratios[0], "center_support_ratio": ratios[1], "plus_support_ratio": ratios[2],
                "connected_surface_ratio": connected, "envelope_pass": envelope_pass,
            }
        )
        ranked.append((key, spec, mesh, masks))
    ranked.sort(key=lambda value: value[0], reverse=True)
    best_key, spec, mesh, masks = ranked[0]
    pd.DataFrame(reports).sort_values(
        ["center_support_ratio", "minus_support_ratio", "plus_support_ratio", "connected_surface_ratio"],
        ascending=False,
        kind="stable",
    ).to_csv(stage / "ellipsoid_search.csv", index=False)
    atomic_write_json(
        stage / "ellipsoid.json",
        {"center_m": spec.center_m.tolist(), "semiaxes_m": spec.semiaxes_m.tolist(), "rotation": spec.rotation.tolist()},
    )
    vertices = pd.DataFrame(
        {
            "vertex_id": np.arange(len(mesh.vertices_m)),
            "unit_x": mesh.unit_vertices[:, 0], "unit_y": mesh.unit_vertices[:, 1], "unit_z": mesh.unit_vertices[:, 2],
            "x_m": mesh.vertices_m[:, 0], "y_m": mesh.vertices_m[:, 1], "z_m": mesh.vertices_m[:, 2],
            "normal_x": mesh.normals[:, 0], "normal_y": mesh.normals[:, 1], "normal_z": mesh.normals[:, 2],
            "minus_supported": masks[0], "center_supported": masks[1], "plus_supported": masks[2],
        }
    )
    faces = pd.DataFrame(
        {
            "face_id": np.arange(len(mesh.faces)),
            "vertex_0": mesh.faces[:, 0], "vertex_1": mesh.faces[:, 1], "vertex_2": mesh.faces[:, 2],
            "area_m2": mesh.face_areas_m2,
        }
    )
    _atomic_parquet(vertices, stage / "mesh_vertices.parquet")
    _atomic_parquet(faces, stage / "mesh_faces.parquet")
    gates = config["gates"]
    return _gate(
        stage / "gate.json",
        {
            "center_support": best_key[0] >= float(gates["center_support_min"]),
            "minus_offset_support": float(np.mean(masks[0])) >= float(gates["offset_support_min"]),
            "plus_offset_support": float(np.mean(masks[2])) >= float(gates["offset_support_min"]),
            "connected_supported_surface": best_key[2] >= float(gates["connected_surface_min"]),
            "registered_mesh_size": (len(mesh.vertices_m), len(mesh.faces))
            in {(42, 80), (642, 1280), (2562, 5120)},
        },
        semantics="full_beta_pool_supported_physical_ellipsoid",
        selected_center_m=spec.center_m.tolist(),
        selected_semiaxes_m=spec.semiaxes_m.tolist(),
        center_support_ratio=best_key[0],
        minus_support_ratio=float(np.mean(masks[0])),
        plus_support_ratio=float(np.mean(masks[2])),
        connected_surface_ratio=best_key[2],
        vertex_count=len(mesh.vertices_m),
        face_count=len(mesh.faces),
    )


def _load_mesh(output_root: Path) -> Any:
    stage = output_root / STAGE_DIR["shell_search"]
    payload = json.loads((stage / "ellipsoid.json").read_text(encoding="utf-8"))
    spec = EllipsoidSpec(payload["center_m"], payload["semiaxes_m"], payload["rotation"])
    vertex_count = len(pd.read_parquet(stage / "mesh_vertices.parquet", columns=["vertex_id"]))
    subdivision_by_count = {42: 1, 642: 3, 2562: 4}
    if vertex_count not in subdivision_by_count:
        raise ValueError(f"unregistered V13 mesh vertex count: {vertex_count}")
    return build_shell_mesh(spec, subdivision_by_count[vertex_count])


def _anchor_frames(config: Mapping[str, Any], project_root: Path) -> dict[str, pd.DataFrame]:
    paths = _sources(config, project_root)
    result: dict[str, pd.DataFrame] = {}
    for chart_id, source_name in (("chart_A", "chart_A_sparse"), ("chart_B", "chart_B_sparse")):
        frame = pd.read_parquet(paths[source_name], columns=[*XYZ_COLUMNS, *BETA_COLUMNS])
        finite = np.isfinite(frame.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS]]).all(axis=1)
        frame = frame.loc[finite].drop_duplicates([*XYZ_COLUMNS, *BETA_COLUMNS]).reset_index(drop=True)
        frame["chart_id"] = chart_id
        result[chart_id] = frame
    return result


def _candidate_policy(config: Mapping[str, Any]) -> CandidatePolicy:
    values = config["candidates"]
    return CandidatePolicy(
        capability_nearest_count=int(values["capability_nearest_count"]),
        capability_representative_count=int(values["capability_representative_count"]),
        candidate_budget_per_node=int(values["candidate_budget_per_node"]),
        difficult_candidate_budget_per_node=int(values["difficult_candidate_budget_per_node"]),
        max_corrector_iterations=int(values["max_corrector_iterations"]),
        max_residual_mm=float(values["max_residual_mm"]),
        gold_margin_deg=float(values["gold_margin_deg"]),
        candidate_cluster_deg=float(values["cluster_deg"]),
    )


def _atlas_policy(config: Mapping[str, Any]) -> AtlasPolicy:
    values = config["atlas"]
    return AtlasPolicy(
        edge_match_deg=float(values["edge_match_deg"]),
        continuation_residual_max_mm=float(config["candidates"]["max_residual_mm"]),
        root_count=int(values["root_count"]),
        top_section_count=int(values["top_section_count"]),
        split_gap_deg=float(values["split_gap_deg"]),
        merge_overlap_p95_deg=float(values["merge_overlap_p95_deg"]),
    )


def _named_anchor_seeds(
    targets: np.ndarray,
    anchors: Mapping[str, pd.DataFrame],
    count: int,
) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {index: {} for index in range(len(targets))}
    for chart_id, frame in anchors.items():
        xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
        beta = frame.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
        tree = cKDTree(xyz)
        _distance, indices = tree.query(targets, k=min(int(count), len(frame)))
        indices = np.asarray(indices, dtype=int)
        if indices.ndim == 1:
            indices = indices[:, None]
        for node_id, selected in enumerate(indices):
            result[node_id][f"{chart_id}_anchor"] = beta[selected]
    return result


def _candidate_frame(bank: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in bank.records:
        rows.append(
            {
                "node_id": record.node_id,
                "candidate_id": record.candidate_id,
                "source": record.source,
                "solver": record.solver,
                "seed_rank": record.seed_rank,
                **{name: float(record.beta_rad[index]) for index, name in enumerate(BETA_COLUMNS)},
                "achieved_x_m": record.achieved_xyz_m[0],
                "achieved_y_m": record.achieved_xyz_m[1],
                "achieved_z_m": record.achieved_xyz_m[2],
                "residual_mm": record.residual_mm,
                "minimum_margin_deg": record.min_margin_deg,
                "normalized_minimum_margin": record.normalized_min_margin,
                "quality_class": record.quality.value,
                "solver_success": record.solver_success,
                "sigma3_m": record.diagnostics.get("sigma3_m", np.nan),
                "kappa": record.diagnostics.get("kappa", np.nan),
                "diagnostics_json": json.dumps(dict(record.diagnostics), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _surface_area_by_chart(mesh: Any, labels: pd.DataFrame) -> dict[str, float]:
    nodes_by_chart = {
        str(chart_id): set(group["vertex_id"].astype(int))
        for chart_id, group in labels.groupby("chart_id", sort=True)
    }
    result: dict[str, float] = {}
    for chart_id, nodes in nodes_by_chart.items():
        covered = np.asarray([all(int(vertex) in nodes for vertex in face) for face in mesh.faces])
        result[chart_id] = float(np.sum(mesh.face_areas_m2[covered]) / mesh.surface_area_m2)
    return result


def stage_surface_atlas(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require_gate(output_root, "shell_search")
    stage = output_root / STAGE_DIR["surface_atlas"]
    stage.mkdir(parents=True, exist_ok=False)
    mesh = _load_mesh(output_root)
    environment = load_environment(project_root, _source_path(project_root, config["robot_config"]))
    pool = _capability_frame(_sources(config, project_root)["capability_pool"])
    capability_xyz = pool.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    capability_beta = pool.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    anchors = _anchor_frames(config, project_root)
    named = _named_anchor_seeds(
        mesh.vertices_m,
        anchors,
        int(config["candidates"]["named_anchor_count_per_chart"]),
    )
    warm_starts: dict[int, np.ndarray] = {}
    warm_provenance: dict[int, list[str]] = {}
    for node_id, banks in named.items():
        ordered_names = sorted(banks)
        warm_starts[node_id] = np.vstack([banks[name] for name in ordered_names])
        warm_provenance[node_id] = [
            f"{name}:{rank:03d}"
            for name in ordered_names
            for rank in range(len(banks[name]))
        ]
    bank = solve_candidate_bank(
        environment,
        mesh.vertices_m,
        _candidate_policy(config),
        capability_beta_rad=capability_beta,
        capability_xyz_m=capability_xyz,
        neighbor_beta_rad=warm_starts,
    )
    candidate_frame = _candidate_frame(bank)
    def seed_provenance(row: pd.Series) -> str:
        source = str(row["source"])
        if not source.startswith("neighbor_warm_start_"):
            return source
        rank = int(source.rsplit("_", 1)[1])
        return warm_provenance[int(row["node_id"])][rank]

    candidate_frame["seed_provenance"] = candidate_frame.apply(
        seed_provenance, axis=1
    )
    _atomic_parquet(candidate_frame, stage / "surface_candidate_bank.parquet")
    task_nodes = tuple(
        AtlasTaskNode(
            node_id=vertex_id,
            xyz_m=mesh.vertices_m[vertex_id],
            neighbor_node_ids=mesh.vertex_neighbors[vertex_id],
        )
        for vertex_id in range(len(mesh.vertices_m))
    )
    candidates: list[AtlasCandidate] = []
    for record in bank.records:
        kappa = float(record.diagnostics.get("kappa", 1.0e12))
        if record.quality is CandidateQuality.REJECT or not np.isfinite(kappa):
            continue
        candidates.append(
            AtlasCandidate(
                node_id=record.node_id,
                candidate_id=record.candidate_id,
                beta_rad=record.beta_rad,
                residual_mm=record.residual_mm,
                min_margin_deg=record.min_margin_deg,
                normalized_min_margin=record.normalized_min_margin,
                condition_number=kappa,
                quality=record.quality.value,
                solver_success=record.solver_success,
                actual_bounds=True,
                diagnostics={"source": record.source, "solver": record.solver},
            )
        )
    atlas = build_canonical_atlas(
        task_nodes,
        candidates,
        make_predictor_corrector_continuation(
            environment,
            max_corrector_iterations=int(config["candidates"]["max_corrector_iterations"]),
            residual_tolerance_mm=float(config["candidates"]["max_residual_mm"]),
        ),
        policy=_atlas_policy(config),
    )
    candidate_by_key = atlas.product_graph.candidate_by_key
    label_rows: list[dict[str, Any]] = []
    chart_rows: list[dict[str, Any]] = []
    for chart in atlas.charts:
        chart_id = f"chart_{chart.chart_id:02d}"
        chart_rows.append(
            {
                "chart_id": chart_id,
                "root_node_id": chart.root_key[0],
                "root_candidate_id": chart.root_key[1],
                "selected_vertex_count": len(chart.selections),
                **{str(key): value for key, value in chart.metrics.items()},
            }
        )
        for vertex_id, candidate_id in chart.selections:
            candidate = candidate_by_key[(vertex_id, candidate_id)]
            label_rows.append(
                {
                    "chart_id": chart_id,
                    "vertex_id": vertex_id,
                    "rho_m": 0.0,
                    "x_m": mesh.vertices_m[vertex_id, 0],
                    "y_m": mesh.vertices_m[vertex_id, 1],
                    "z_m": mesh.vertices_m[vertex_id, 2],
                    **{name: float(candidate.beta_rad[index]) for index, name in enumerate(BETA_COLUMNS)},
                    "quality_class": candidate.quality,
                    "residual_mm": candidate.residual_mm,
                    "minimum_margin_deg": candidate.min_margin_deg,
                    "candidate_gap_p95_deg": 0.0,
                    "candidate_gap_max_deg": 0.0,
                    "successful_parent_count": 1,
                    "source_candidate_id": candidate_id,
                }
            )
    labels = pd.DataFrame(label_rows)
    charts = pd.DataFrame(chart_rows)
    _atomic_parquet(labels, stage / "surface_labels.parquet")
    charts.to_csv(stage / "charts.csv", index=False)
    overlap_rows = [
        {
            "chart_a_id": f"chart_{value.chart_a_id:02d}",
            "chart_b_id": f"chart_{value.chart_b_id:02d}",
            "shared_vertex_count": len(value.shared_node_ids),
            "gap_p95_deg": value.gap_p95_deg,
            "gap_max_deg": value.gap_max_deg,
            "resolution": value.resolution,
        }
        for value in atlas.overlap_reports
    ]
    pd.DataFrame(overlap_rows).to_csv(stage / "chart_overlaps.csv", index=False)
    area_by_chart = _surface_area_by_chart(mesh, labels) if len(labels) else {}
    chart_sets = [set(group["vertex_id"].astype(int)) for _, group in labels.groupby("chart_id", sort=True)]
    union_faces = np.asarray(
        [any(all(int(vertex) in chart_nodes for vertex in face) for chart_nodes in chart_sets) for face in mesh.faces]
    )
    union_area = float(np.sum(mesh.face_areas_m2[union_faces]) / mesh.surface_area_m2)
    mergeable_overlap = [
        value
        for value in atlas.overlap_reports
        if value.resolution.startswith("mergeable")
    ]
    ambiguous_overlap = [
        value
        for value in atlas.overlap_reports
        if value.resolution == "ambiguous_keep_separate"
    ]
    gates = config["gates"]
    return _gate(
        stage / "gate.json",
        {
            "surface_candidates_exist": len(candidates) > 0,
            "canonical_chart_exists": len(atlas.charts) > 0,
            "largest_chart_surface": max(area_by_chart.values(), default=0.0) >= float(gates["largest_chart_surface_min"]),
            "surface_union": union_area >= float(gates["surface_union_min"]),
            "mergeable_overlap_p95": all(value.gap_p95_deg <= float(config["atlas"]["merge_overlap_p95_deg"]) + 1.0e-12 for value in mergeable_overlap),
            "mergeable_overlap_max": all(value.gap_max_deg <= float(config["atlas"]["merge_overlap_max_deg"]) + 1.0e-12 for value in mergeable_overlap),
            "automatic_chart_classifier_disabled": config["student"]["automatic_chart_classifier"] is False,
        },
        semantics="bidirectional_surface_product_graph_and_known_charts",
        chart_count=len(atlas.charts),
        area_ratio_by_chart=area_by_chart,
        surface_union_ratio=union_area,
        robust_edge_count=len(atlas.product_graph.robust_edges),
        directed_edge_count=len(atlas.product_graph.directed_edges),
        rejected_continuation_count=atlas.product_graph.rejected_continuation_count,
        ambiguous_known_chart_overlap_count=len(ambiguous_overlap),
    )


def _region_policy(mesh: Any, config: Mapping[str, Any]) -> RegionLabelPolicy:
    maximum_edge_mm = max(
        float(np.linalg.norm(mesh.vertices_m[left] - mesh.vertices_m[right]) * 1000.0)
        for left, neighbors in enumerate(mesh.vertex_neighbors)
        for right in neighbors
    )
    return RegionLabelPolicy(
        parent_distance_max_mm=max(25.0, maximum_edge_mm + 25.0),
        parent_attempt_count=5,
        successful_parent_minimum=3,
        continuation_substep_mm=5.0,
        gold_residual_max_mm=1.0,
        gold_candidate_gap_max_deg=0.5,
        silver_residual_max_mm=float(config["candidates"]["max_residual_mm"]),
        silver_candidate_gap_max_deg=1.0,
        edge_distance_max_mm=max(10.0, maximum_edge_mm),
        teacher_policy=TeacherPolicy(
            tracking_tolerance_mm=float(config["candidates"]["max_residual_mm"]),
            max_corrector_iterations=int(config["candidates"]["max_corrector_iterations"]),
            safe_margin_repulsion_step_deg=0.0,
            beta_weights=(4.0, 4.0, 2.0, 2.0, 1.0, 1.0),
        ),
    )


def _label_key(chart_id: str, rho_m: float, vertex_id: int) -> tuple[str, float, int]:
    return str(chart_id), round(float(rho_m), 9), int(vertex_id)


def _label_beta(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([row[name] for name in BETA_COLUMNS], dtype=float)


def _label_one_radial_target(
    environment: Any,
    target_xyz: np.ndarray,
    parents: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    policy: RegionLabelPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for source_name, parent in parents[: policy.parent_attempt_count]:
        outcome = continue_from_parent(
            environment,
            target_xyz,
            np.asarray([parent[name] for name in XYZ_COLUMNS], dtype=float),
            _label_beta(parent),
            policy=policy,
        )
        attempts.append({"parent_source": source_name, **outcome})
    frame = pd.DataFrame(attempts)
    for name in ("success", "residual_mm", *BETA_COLUMNS):
        if name not in frame:
            frame[name] = False if name == "success" else np.nan
    reduced = reduce_parent_candidates(frame, policy=policy, required_successes=3)
    return reduced, attempts


def _margin_and_condition(environment: Any, beta: np.ndarray) -> tuple[float, float, float]:
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    margin = float(np.rad2deg(np.min(np.minimum(beta - bounds[:, 0], bounds[:, 1] - beta))))
    singular = np.linalg.svd(np.asarray(environment.jacobian(beta), dtype=float).reshape(3, 6), compute_uv=False)
    return margin, float(singular[-1]), float(singular[0] / max(singular[-1], 1.0e-12))


def stage_radial_labels(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require_gate(output_root, "surface_atlas")
    stage = output_root / STAGE_DIR["radial_labels"]
    stage.mkdir(parents=True, exist_ok=False)
    mesh = _load_mesh(output_root)
    environment = load_environment(project_root, _source_path(project_root, config["robot_config"]))
    policy = _region_policy(mesh, config)
    surface = pd.read_parquet(output_root / STAGE_DIR["surface_atlas"] / "surface_labels.parquet")
    labels: dict[tuple[str, float, int], dict[str, Any]] = {}
    for row in surface.to_dict(orient="records"):
        labels[_label_key(row["chart_id"], 0.0, row["vertex_id"])] = dict(row)
    levels = np.asarray(config["mesh"]["radial_levels_mm"], dtype=float) / 1000.0
    nonzero = sorted((float(value) for value in levels if not np.isclose(value, 0.0)), key=lambda value: (abs(value), value))
    attempt_rows: list[dict[str, Any]] = []
    chart_ids = sorted(surface["chart_id"].astype(str).unique())
    for rho in nonzero:
        previous_candidates = [value for value in levels if value * rho > 0.0 and abs(value) < abs(rho)]
        previous_rho = float(max(previous_candidates, key=abs)) if previous_candidates else 0.0
        pending: list[tuple[str, int]] = []
        for chart_id in chart_ids:
            center_vertices = sorted(
                int(value) for value in surface.loc[surface["chart_id"].eq(chart_id), "vertex_id"].unique()
            )
            for vertex_id in center_vertices:
                target = mesh.offset_vertices(rho)[vertex_id]
                center_row = labels[_label_key(chart_id, 0.0, vertex_id)]
                radial_row = labels.get(_label_key(chart_id, previous_rho, vertex_id))
                parents: list[tuple[str, Mapping[str, Any]]] = []
                if radial_row is not None:
                    parents.append(("radial_parent", radial_row))
                parents.append(("surface_center", center_row))
                for neighbor_id in mesh.vertex_neighbors[vertex_id]:
                    neighbor = labels.get(_label_key(chart_id, previous_rho, neighbor_id))
                    if neighbor is not None:
                        parents.append((f"previous_layer_neighbor_{neighbor_id}", neighbor))
                reduced, attempts = _label_one_radial_target(environment, target, parents, policy=policy)
                for attempt in attempts:
                    attempt_rows.append({"chart_id": chart_id, "rho_m": rho, "vertex_id": vertex_id, "pass": 1, **attempt})
                if reduced.get("accepted"):
                    beta = np.asarray([reduced[name] for name in BETA_COLUMNS], dtype=float)
                    margin, sigma3, kappa = _margin_and_condition(environment, beta)
                    labels[_label_key(chart_id, rho, vertex_id)] = {
                        "chart_id": chart_id,
                        "vertex_id": vertex_id,
                        "rho_m": rho,
                        "x_m": target[0], "y_m": target[1], "z_m": target[2],
                        **{name: float(beta[index]) for index, name in enumerate(BETA_COLUMNS)},
                        **{key: value for key, value in reduced.items() if key != "selected_candidate_row" and not key.endswith("_rows")},
                        "residual_mm": float(reduced["residual_max_mm"]),
                        "minimum_margin_deg": margin,
                        "sigma3_m": sigma3,
                        "kappa": kappa,
                        "label_pass": 1,
                    }
                else:
                    pending.append((chart_id, vertex_id))
        # Second pass adds already accepted same-layer neighbors, matching the
        # registered two-pass radial-label rule without order-dependent labels.
        for chart_id, vertex_id in pending:
            target = mesh.offset_vertices(rho)[vertex_id]
            parents = []
            radial_row = labels.get(_label_key(chart_id, previous_rho, vertex_id))
            if radial_row is not None:
                parents.append(("radial_parent", radial_row))
            parents.append(("surface_center", labels[_label_key(chart_id, 0.0, vertex_id)]))
            for neighbor_id in mesh.vertex_neighbors[vertex_id]:
                neighbor = labels.get(_label_key(chart_id, rho, neighbor_id))
                if neighbor is not None:
                    parents.append((f"same_layer_neighbor_{neighbor_id}", neighbor))
            reduced, attempts = _label_one_radial_target(environment, target, parents, policy=policy)
            for attempt in attempts:
                attempt_rows.append({"chart_id": chart_id, "rho_m": rho, "vertex_id": vertex_id, "pass": 2, **attempt})
            if reduced.get("accepted"):
                beta = np.asarray([reduced[name] for name in BETA_COLUMNS], dtype=float)
                margin, sigma3, kappa = _margin_and_condition(environment, beta)
                labels[_label_key(chart_id, rho, vertex_id)] = {
                    "chart_id": chart_id,
                    "vertex_id": vertex_id,
                    "rho_m": rho,
                    "x_m": target[0], "y_m": target[1], "z_m": target[2],
                    **{name: float(beta[index]) for index, name in enumerate(BETA_COLUMNS)},
                    **{key: value for key, value in reduced.items() if key != "selected_candidate_row" and not key.endswith("_rows")},
                    "residual_mm": float(reduced["residual_max_mm"]),
                    "minimum_margin_deg": margin,
                    "sigma3_m": sigma3,
                    "kappa": kappa,
                    "label_pass": 2,
                }
    label_frame = pd.DataFrame(labels.values()).sort_values(["chart_id", "rho_m", "vertex_id"], kind="stable")
    _atomic_parquet(label_frame, stage / "radial_labels.parquet")
    _atomic_parquet(pd.DataFrame(attempt_rows), stage / "radial_parent_attempts.parquet")
    chart_sets = np.empty((len(levels), len(mesh.vertices_m)), dtype=object)
    for level_index in range(len(levels)):
        for vertex_id in range(len(mesh.vertices_m)):
            chart_sets[level_index, vertex_id] = set()
    level_index_by_rho = {round(float(value), 9): index for index, value in enumerate(levels)}
    for row in label_frame.itertuples(index=False):
        chart_sets[level_index_by_rho[round(float(row.rho_m), 9)], int(row.vertex_id)].add(str(row.chart_id))
    coverage = shell_coverage_report(mesh, levels, chart_sets)
    gaps = label_frame.loc[label_frame["rho_m"].ne(0.0), "candidate_gap_max_deg"].dropna().to_numpy(dtype=float)
    gates = config["gates"]
    quantiles_mm = {name: value * 1000.0 for name, value in coverage.thickness_area_weighted_quantiles_m.items()}
    return _gate(
        stage / "gate.json",
        {
            "same_chart_surface_union": coverage.surface_area_ratio >= float(gates["surface_union_min"]),
            "same_chart_shell_volume": coverage.volume_ratio >= float(gates["shell_volume_min"]),
            "thickness_p10": quantiles_mm["p10"] >= float(gates["thickness_p10_min_mm"]),
            "thickness_p50": quantiles_mm["p50"] >= float(gates["thickness_p50_min_mm"]),
            "radial_parent_consistency_p95": len(gaps) > 0 and float(np.percentile(gaps, 95)) <= 0.5 + 1.0e-12,
            "radial_parent_consistency_max": len(gaps) > 0 and float(np.max(gaps)) <= 1.0 + 1.0e-12,
            "radial_residual_max": len(label_frame) > 0 and float(label_frame["residual_mm"].fillna(0.0).max()) <= float(config["candidates"]["max_residual_mm"]) + 1.0e-12,
            "all_labels_in_bounds": bool(
                np.all(label_frame.loc[:, BETA_COLUMNS].to_numpy() >= np.asarray(environment.bounds)[:, 0] - 1.0e-12)
                and np.all(label_frame.loc[:, BETA_COLUMNS].to_numpy() <= np.asarray(environment.bounds)[:, 1] + 1.0e-12)
            ),
        },
        semantics="same_chart_multi_parent_physical_radial_shell",
        surface_area_ratio=coverage.surface_area_ratio,
        shell_volume_ratio=coverage.volume_ratio,
        surface_area_total_m2=coverage.surface_area_total_m2,
        shell_volume_total_m3=coverage.shell_volume_total_m3,
        thickness_area_weighted_quantiles_mm=quantiles_mm,
        radial_parent_gap_p95_deg=float(np.percentile(gaps, 95)) if len(gaps) else None,
        radial_parent_gap_max_deg=float(np.max(gaps)) if len(gaps) else None,
        accepted_label_rows=len(label_frame),
    )


def _radial_chart_arrays(
    mesh: Any,
    levels: np.ndarray,
    labels: pd.DataFrame,
) -> tuple[np.ndarray, dict[tuple[int, int, str], np.ndarray]]:
    chart_sets = np.empty((len(levels), len(mesh.vertices_m)), dtype=object)
    for level_index in range(len(levels)):
        for vertex_id in range(len(mesh.vertices_m)):
            chart_sets[level_index, vertex_id] = set()
    level_by_rho = {round(float(value), 9): index for index, value in enumerate(levels)}
    beta_by_key: dict[tuple[int, int, str], np.ndarray] = {}
    for row in labels.itertuples(index=False):
        level_index = level_by_rho[round(float(row.rho_m), 9)]
        vertex_id = int(row.vertex_id)
        chart_id = str(row.chart_id)
        chart_sets[level_index, vertex_id].add(chart_id)
        beta_by_key[(level_index, vertex_id, chart_id)] = np.asarray(
            [getattr(row, name) for name in BETA_COLUMNS], dtype=float
        )
    return chart_sets, beta_by_key


def _parent_prediction_metrics(
    environment: Any,
    mesh: Any,
    levels: np.ndarray,
    samples: Mapping[str, np.ndarray],
    chart_ids: np.ndarray,
    beta_by_key: Mapping[tuple[int, int, str], np.ndarray],
    *,
    chunk_rows: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare six independent local parent predictions at each dense point."""
    count = len(chart_ids)
    gap_p95 = np.empty(count, dtype=float)
    gap_max = np.empty(count, dtype=float)
    in_bounds_count = np.empty(count, dtype=np.int16)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    pinv_by_key: dict[tuple[int, int, str], np.ndarray] = {}
    xyz_by_key: dict[tuple[int, int], np.ndarray] = {}
    for start in range(0, count, int(chunk_rows)):
        stop = min(count, start + int(chunk_rows))
        predictions = np.empty((stop - start, 6, 6), dtype=float)
        for local_row, sample_id in enumerate(range(start, stop)):
            radial_index = int(samples["radial_interval_id"][sample_id])
            face = mesh.faces[int(samples["face_id"][sample_id])]
            chart_id = str(chart_ids[sample_id])
            endpoint_keys = [
                (radial_index, int(face[0]), chart_id),
                (radial_index, int(face[1]), chart_id),
                (radial_index, int(face[2]), chart_id),
                (radial_index + 1, int(face[0]), chart_id),
                (radial_index + 1, int(face[1]), chart_id),
                (radial_index + 1, int(face[2]), chart_id),
            ]
            target = samples["xyz_m"][sample_id]
            for parent_id, key in enumerate(endpoint_keys):
                beta = beta_by_key[key]
                if key not in pinv_by_key:
                    jacobian = np.asarray(environment.jacobian(beta), dtype=float).reshape(3, 6)
                    pinv_by_key[key] = weighted_damped_pinv(
                        jacobian,
                        damping=1.0e-3,
                        weights=(4.0, 4.0, 2.0, 2.0, 1.0, 1.0),
                    )
                xyz_key = (key[0], key[1])
                if xyz_key not in xyz_by_key:
                    xyz_by_key[xyz_key] = mesh.offset_vertices(float(levels[key[0]]))[key[1]]
                predictions[local_row, parent_id] = beta + pinv_by_key[key] @ (
                    target - xyz_by_key[xyz_key]
                )
        pairwise = np.rad2deg(
            np.sqrt(
                np.mean(
                    np.square(
                        predictions[:, :, None, :]
                        - predictions[:, None, :, :]
                    ),
                    axis=-1,
                )
            )
        )
        gap_p95[start:stop] = np.percentile(pairwise, 95, axis=(1, 2))
        gap_max[start:stop] = np.max(pairwise, axis=(1, 2))
        in_bounds_count[start:stop] = np.sum(
            np.all(predictions >= bounds[:, 0].reshape(1, 1, 6) - 1.0e-12, axis=2)
            & np.all(predictions <= bounds[:, 1].reshape(1, 1, 6) + 1.0e-12, axis=2),
            axis=1,
        )
    return gap_p95, gap_max, in_bounds_count


def _volume_stratified_accepted_indices(
    *,
    accepted: np.ndarray,
    radial_interval_id: np.ndarray,
    face_id: np.ndarray,
    accepted_cell_mask: np.ndarray,
    cell_volumes: np.ndarray,
    requested_count: int,
) -> np.ndarray:
    """Select accepted rows with deterministic physical-volume cell quotas."""
    accepted_mask = np.asarray(accepted, dtype=bool).reshape(-1)
    radial = np.asarray(radial_interval_id, dtype=np.int64).reshape(-1)
    faces = np.asarray(face_id, dtype=np.int64).reshape(-1)
    if len(radial) != len(accepted_mask) or len(faces) != len(accepted_mask):
        raise ValueError("accepted and cell identifiers must align")
    mask = np.asarray(accepted_cell_mask, dtype=bool)
    volumes = np.asarray(cell_volumes, dtype=float)
    if mask.shape != volumes.shape:
        raise ValueError("accepted cell mask and volume table must align")
    accepted_rows = np.flatnonzero(accepted_mask)
    if len(accepted_rows) <= int(requested_count):
        return accepted_rows
    cell_rows = np.argwhere(mask)
    weights = volumes[mask]
    expected = int(requested_count) * weights / weights.sum()
    quota = np.floor(expected).astype(np.int64)
    remainder = int(requested_count) - int(quota.sum())
    fractional = expected - quota
    order = np.lexsort((np.arange(len(quota)), -fractional))
    quota[order[:remainder]] += 1
    face_count = mask.shape[1]
    cell_codes = radial * face_count + faces
    accepted_codes = cell_codes[accepted_rows]
    stable_order = np.argsort(accepted_codes, kind="stable")
    sorted_rows = accepted_rows[stable_order]
    sorted_codes = accepted_codes[stable_order]
    unique, starts, counts = np.unique(
        sorted_codes, return_index=True, return_counts=True
    )
    row_groups = {
        int(code): sorted_rows[int(start) : int(start + count)]
        for code, start, count in zip(unique, starts, counts)
    }
    capacity = np.asarray(
        [
            len(row_groups.get(int(r * face_count + f), ()))
            for r, f in cell_rows
        ],
        dtype=np.int64,
    )
    quota = np.minimum(quota, capacity)
    deficit = int(requested_count) - int(quota.sum())
    while deficit > 0:
        eligible = np.flatnonzero(quota < capacity)
        if not len(eligible):
            break
        priority = sorted(
            eligible,
            key=lambda index: (
                float(quota[index] / max(expected[index], 1.0e-12)),
                int(index),
            ),
        )
        take = priority[:deficit]
        quota[np.asarray(take, dtype=int)] += 1
        deficit -= len(take)
    selected: list[np.ndarray] = []
    for (radial_index, selected_face), count in zip(cell_rows, quota):
        if count:
            code = int(radial_index * face_count + selected_face)
            selected.append(row_groups[code][: int(count)])
    return np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)


def stage_dense_dataset(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require_gate(output_root, "radial_labels")
    stage = output_root / STAGE_DIR["dense_dataset"]
    stage.mkdir(parents=True, exist_ok=False)
    mesh = _load_mesh(output_root)
    levels = np.asarray(config["mesh"]["radial_levels_mm"], dtype=float) / 1000.0
    labels = pd.read_parquet(output_root / STAGE_DIR["radial_labels"] / "radial_labels.parquet")
    chart_sets, beta_by_key = _radial_chart_arrays(mesh, levels, labels)
    cell_mask, primary_chart = covered_cells(mesh, levels, chart_sets)
    requested = int(config["dense"]["row_count"])
    attempt_count = requested * int(config["dense"]["attempt_multiplier"])
    samples = sample_shell_cells(
        mesh,
        levels,
        cell_mask,
        attempt_count,
        seed=int(config["dense"]["seed"]),
        stratified=True,
    )
    chart_ids = np.asarray(
        [
            primary_chart[int(radial), int(face)]
            for radial, face in zip(samples["radial_interval_id"], samples["face_id"])
        ],
        dtype=object,
    )
    beta = np.empty((attempt_count, 6), dtype=float)
    overlap_count = np.empty(attempt_count, dtype=np.int16)
    parent_ids: list[str] = []
    for sample_id in range(attempt_count):
        radial_index = int(samples["radial_interval_id"][sample_id])
        face_id = int(samples["face_id"][sample_id])
        face = mesh.faces[face_id]
        chart_id = str(chart_ids[sample_id])
        keys = [
            (radial_index, int(face[0]), chart_id),
            (radial_index, int(face[1]), chart_id),
            (radial_index, int(face[2]), chart_id),
            (radial_index + 1, int(face[0]), chart_id),
            (radial_index + 1, int(face[1]), chart_id),
            (radial_index + 1, int(face[2]), chart_id),
        ]
        beta[sample_id] = samples["prism_vertex_weights"][sample_id] @ np.vstack(
            [beta_by_key[key] for key in keys]
        )
        common = set(chart_sets[radial_index, int(face[0])])
        for level_index in (radial_index, radial_index + 1):
            for vertex_id in face:
                common &= set(chart_sets[level_index, int(vertex_id)])
        overlap_count[sample_id] = len(common)
        parent_ids.append(
            json.dumps([f"L{key[0]}:V{key[1]}:{key[2]}" for key in keys])
        )
    environment = load_environment(
        project_root, _source_path(project_root, config["robot_config"])
    )
    achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
    residual_mm = np.linalg.norm(achieved - samples["xyz_m"], axis=1) * 1000.0
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    actual_bounds = np.all(beta >= bounds[:, 0] - 1.0e-12, axis=1) & np.all(
        beta <= bounds[:, 1] + 1.0e-12, axis=1
    )
    parent_gap_p95, parent_gap_max, successful_parent_count = _parent_prediction_metrics(
        environment, mesh, levels, samples, chart_ids, beta_by_key
    )
    dense_values = config["dense"]
    accepted = (
        actual_bounds
        & (residual_mm <= float(dense_values["residual_max_mm"]) + 1.0e-12)
        & (successful_parent_count >= 3)
        & (parent_gap_max <= float(dense_values["parent_gap_max_deg"]) + 1.0e-12)
    )
    volumes = radial_cell_volumes(mesh, levels)
    accepted_indices = _volume_stratified_accepted_indices(
        accepted=accepted,
        radial_interval_id=samples["radial_interval_id"],
        face_id=samples["face_id"],
        accepted_cell_mask=cell_mask,
        cell_volumes=volumes,
        requested_count=requested,
    )
    surface_unit = np.empty((attempt_count, 3), dtype=float)
    normal = np.empty((attempt_count, 3), dtype=float)
    for sample_id in range(attempt_count):
        face = mesh.faces[int(samples["face_id"][sample_id])]
        bary = samples["surface_barycentric"][sample_id]
        unit = bary @ mesh.unit_vertices[face]
        surface_unit[sample_id] = unit / np.linalg.norm(unit)
        local_normal = bary @ mesh.normals[face]
        normal[sample_id] = local_normal / np.linalg.norm(local_normal)
    cell_code = samples["radial_interval_id"] * len(mesh.faces) + samples["face_id"]
    split_code = (samples["face_id"] * 131 + samples["radial_interval_id"] * 17) % 20
    split = np.where(
        split_code < 14,
        "train",
        np.where(split_code < 17, "validation", "sealed"),
    )
    quality = np.where(
        (residual_mm <= 1.0) & (parent_gap_max <= 0.5),
        "RegionGold",
        "RegionSilver",
    )
    frame = pd.DataFrame(
        {
            "shell_id": "ellipsoid_00",
            "chart_id": chart_ids,
            "surface_id": 0,
            "face_id": samples["face_id"],
            "radial_interval_id": samples["radial_interval_id"],
            "cell_id": cell_code,
            "unit_x": surface_unit[:, 0],
            "unit_y": surface_unit[:, 1],
            "unit_z": surface_unit[:, 2],
            "barycentric_0": samples["surface_barycentric"][:, 0],
            "barycentric_1": samples["surface_barycentric"][:, 1],
            "barycentric_2": samples["surface_barycentric"][:, 2],
            "rho_m": samples["rho_m"],
            "normal_x": normal[:, 0],
            "normal_y": normal[:, 1],
            "normal_z": normal[:, 2],
            "x_m": samples["xyz_m"][:, 0],
            "y_m": samples["xyz_m"][:, 1],
            "z_m": samples["xyz_m"][:, 2],
            **{name: beta[:, index] for index, name in enumerate(BETA_COLUMNS)},
            "parent_ids_json": parent_ids,
            "successful_parent_count": successful_parent_count,
            "candidate_gap_p95_deg": parent_gap_p95,
            "candidate_gap_max_deg": parent_gap_max,
            "residual_mm": residual_mm,
            "actual_bounds": actual_bounds,
            "quality_class": quality,
            "overlap_chart_count": overlap_count,
            "split": split,
            "accepted": accepted,
        }
    )
    _atomic_parquet(frame, stage / "dense_attempts.parquet")
    dataset = frame.iloc[accepted_indices].copy().reset_index(drop=True)
    dataset.insert(0, "sample_id", np.arange(len(dataset), dtype=np.int64))
    _atomic_parquet(dataset, stage / "A3_shell_dataset.parquet")
    accepted_counts = dataset.groupby(["radial_interval_id", "face_id"], sort=True).size()
    density = []
    for radial_index, face_id in np.argwhere(cell_mask):
        count = int(accepted_counts.get((int(radial_index), int(face_id)), 0))
        density.append(count / volumes[int(radial_index), int(face_id)])
    density = np.asarray(density, dtype=float)
    density_cv = (
        float(np.std(density) / np.mean(density))
        if len(density) and np.mean(density) > 0.0
        else math.inf
    )
    if len(dataset):
        probes = sample_shell_cells(
            mesh,
            levels,
            cell_mask,
            min(10000, max(1000, len(dataset))),
            seed=int(config["dense"]["seed"]) + 1,
            stratified=True,
        )["xyz_m"]
        distance, _ = cKDTree(dataset.loc[:, XYZ_COLUMNS].to_numpy()).query(
            probes, k=1
        )
        fill_p95_mm = float(np.percentile(distance, 95) * 1000.0)
        fill_max_mm = float(np.max(distance) * 1000.0)
    else:
        fill_p95_mm = fill_max_mm = math.inf
    gap_values = (
        dataset["candidate_gap_max_deg"].to_numpy(dtype=float)
        if len(dataset)
        else np.asarray([])
    )
    return _gate(
        stage / "gate.json",
        {
            "requested_rows_written_without_padding": len(dataset) == requested,
            "physical_volume_density_cv": density_cv <= float(dense_values["density_cv_max"]) + 1.0e-12,
            "fill_distance_p95": fill_p95_mm <= float(dense_values["fill_distance_p95_mm"]) + 1.0e-12,
            "fill_distance_max": fill_max_mm <= float(dense_values["fill_distance_max_mm"]) + 1.0e-12,
            "parent_consistency_p95": len(gap_values) > 0 and float(np.percentile(gap_values, 95)) <= float(dense_values["parent_gap_p95_deg"]) + 1.0e-12,
            "parent_consistency_max": len(gap_values) > 0 and float(np.max(gap_values)) <= float(dense_values["parent_gap_max_deg"]) + 1.0e-12,
            "fk_residual_max": len(dataset) > 0 and float(dataset["residual_mm"].max()) <= float(dense_values["residual_max_mm"]) + 1.0e-12,
            "all_beta_in_bounds": len(dataset) > 0 and bool(dataset["actual_bounds"].all()),
            "known_chart_only": len(dataset) > 0 and bool(dataset["chart_id"].astype(str).str.len().gt(0).all()),
        },
        semantics="uniform_accepted_prism_volume_with_consistent_known_chart_labels",
        requested_rows=requested,
        accepted_attempt_rows=int(np.sum(accepted)),
        written_rows=len(dataset),
        acceptance_ratio=float(np.mean(accepted)),
        density_cv=density_cv,
        fill_distance_p95_mm=fill_p95_mm,
        fill_distance_max_mm=fill_max_mm,
        parent_gap_p95_deg=float(np.percentile(gap_values, 95)) if len(gap_values) else None,
        parent_gap_max_deg=float(np.max(gap_values)) if len(gap_values) else None,
        residual_p95_mm=float(np.percentile(dataset["residual_mm"], 95)) if len(dataset) else None,
        residual_max_mm=float(dataset["residual_mm"].max()) if len(dataset) else None,
        split_rows=dataset["split"].value_counts().sort_index().to_dict(),
        chart_rows=dataset["chart_id"].value_counts().sort_index().to_dict(),
    )


def stage_summary(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require_gate(output_root, "dense_dataset")
    stage = output_root / STAGE_DIR["summary"]
    stage.mkdir(parents=True, exist_ok=False)
    gates = {
        name: json.loads(
            (output_root / directory / "gate.json").read_text(encoding="utf-8")
        )
        for name, directory in STAGES[:-1]
    }
    dataset_path = output_root / STAGE_DIR["dense_dataset"] / "A3_shell_dataset.parquet"
    dataset_rows = len(pd.read_parquet(dataset_path, columns=["sample_id"]))
    source_manifest = json.loads(
        (output_root / STAGE_DIR["protocol"] / "source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return _gate(
        stage / "gate.json",
        {
            "all_primary_shell_stages_pass": all(
                bool(value["gate_pass"]) for value in gates.values()
            ),
            "dataset_exists": dataset_path.is_file(),
            "dataset_rows_match": dataset_rows == int(config["dense"]["row_count"]),
            "student_not_used_to_upgrade_shell": True,
            "automatic_chart_classifier_disabled": config["student"]["automatic_chart_classifier"] is False,
        },
        semantics="primary_shell_dataset_claim_before_student",
        source_git_sha=source_manifest["git_sha"],
        dataset_path=str(dataset_path),
        dataset_sha256=sha256_file(dataset_path),
        dataset_rows=dataset_rows,
        stage_gate_pass={name: bool(value["gate_pass"]) for name, value in gates.items()},
        student_training_status="not_started_by_primary_shell_pipeline",
    )


def run_dense_quota_replay(
    config: Mapping[str, Any],
    project_root: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Replay only the deterministic accepted-row quota from frozen attempts."""
    if output_root.exists():
        raise FileExistsError(f"dense replay output already exists: {output_root}")
    output_root.mkdir(parents=True)
    protocol = output_root / STAGE_DIR["protocol"]
    protocol.mkdir()
    source_gates = {
        name: json.loads(
            (source_root / STAGE_DIR[name] / "gate.json").read_text(encoding="utf-8")
        )
        for name in ("protocol", "shell_search", "surface_atlas", "radial_labels", "dense_dataset")
    }
    dense_checks = dict(source_gates["dense_dataset"]["checks"])
    expected_failure = {
        name for name, value in dense_checks.items() if not bool(value)
    } == {"physical_volume_density_cv"}
    source_files = {
        "ellipsoid": source_root / STAGE_DIR["shell_search"] / "ellipsoid.json",
        "mesh_vertices": source_root / STAGE_DIR["shell_search"] / "mesh_vertices.parquet",
        "mesh_faces": source_root / STAGE_DIR["shell_search"] / "mesh_faces.parquet",
        "radial_labels": source_root / STAGE_DIR["radial_labels"] / "radial_labels.parquet",
        "dense_attempts": source_root / STAGE_DIR["dense_dataset"] / "dense_attempts.parquet",
    }
    git_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "replay_git_sha": git_sha,
        "source_run_root": str(source_root),
        "source_git_sha": source_gates["protocol"].get("git_sha"),
        "source_artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_files.items()
        },
        "replay_implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_write_json(protocol / "source_manifest.json", manifest)
    protocol_gate = _gate(
        protocol / "gate.json",
        {
            "source_files_exist": all(path.is_file() for path in source_files.values()),
            "source_upstream_shell_gates_pass": all(
                bool(source_gates[name]["gate_pass"])
                for name in ("protocol", "shell_search", "surface_atlas", "radial_labels")
            ),
            "source_dense_failure_is_quota_cv_only": expected_failure,
            "replay_worktree_clean": git_status == "",
            "formal_preset": config["preset"] == "formal",
        },
        semantics="frozen_formal_attempts_dense_quota_replay",
        **manifest,
    )
    if not protocol_gate["gate_pass"]:
        report = {
            "protocol_id": PROTOCOL_ID,
            "preset": config["preset"],
            "output_root": str(output_root),
            "requested_stage": "dense_quota_replay",
            "executed_stages": ["protocol"],
            "stopped_after": "protocol",
        }
        atomic_write_json(output_root / "run_report.json", report)
        return report
    mesh = _load_mesh(source_root)
    levels = np.asarray(config["mesh"]["radial_levels_mm"], dtype=float) / 1000.0
    labels = pd.read_parquet(source_files["radial_labels"])
    chart_sets, _beta_by_key = _radial_chart_arrays(mesh, levels, labels)
    cell_mask, _primary = covered_cells(mesh, levels, chart_sets)
    volumes = radial_cell_volumes(mesh, levels)
    attempts = pd.read_parquet(source_files["dense_attempts"])
    requested = int(config["dense"]["row_count"])
    selected = _volume_stratified_accepted_indices(
        accepted=attempts["accepted"].to_numpy(dtype=bool),
        radial_interval_id=attempts["radial_interval_id"].to_numpy(),
        face_id=attempts["face_id"].to_numpy(),
        accepted_cell_mask=cell_mask,
        cell_volumes=volumes,
        requested_count=requested,
    )
    dataset = attempts.iloc[selected].copy().reset_index(drop=True)
    dataset.insert(0, "sample_id", np.arange(len(dataset), dtype=np.int64))
    dense_stage = output_root / STAGE_DIR["dense_dataset"]
    dense_stage.mkdir(parents=True)
    dataset_path = dense_stage / "A3_shell_dataset.parquet"
    _atomic_parquet(dataset, dataset_path)
    counts = dataset.groupby(["radial_interval_id", "face_id"], sort=True).size()
    density = np.asarray(
        [
            int(counts.get((int(radial), int(face)), 0))
            / volumes[int(radial), int(face)]
            for radial, face in np.argwhere(cell_mask)
        ]
    )
    density_cv = float(np.std(density) / np.mean(density))
    probes = sample_shell_cells(
        mesh,
        levels,
        cell_mask,
        10000,
        seed=int(config["dense"]["seed"]) + 1,
        stratified=True,
    )["xyz_m"]
    distance, _ = cKDTree(dataset.loc[:, XYZ_COLUMNS].to_numpy()).query(probes, k=1)
    fill_p95_mm = float(np.percentile(distance, 95) * 1000.0)
    fill_max_mm = float(np.max(distance) * 1000.0)
    dense_values = config["dense"]
    gap = dataset["candidate_gap_max_deg"].to_numpy(dtype=float)
    dense_gate = _gate(
        dense_stage / "gate.json",
        {
            "requested_rows_written_without_padding": len(dataset) == requested,
            "physical_volume_density_cv": density_cv <= float(dense_values["density_cv_max"]) + 1.0e-12,
            "fill_distance_p95": fill_p95_mm <= float(dense_values["fill_distance_p95_mm"]) + 1.0e-12,
            "fill_distance_max": fill_max_mm <= float(dense_values["fill_distance_max_mm"]) + 1.0e-12,
            "parent_consistency_p95": float(np.percentile(gap, 95)) <= float(dense_values["parent_gap_p95_deg"]) + 1.0e-12,
            "parent_consistency_max": float(np.max(gap)) <= float(dense_values["parent_gap_max_deg"]) + 1.0e-12,
            "fk_residual_max": float(dataset["residual_mm"].max()) <= float(dense_values["residual_max_mm"]) + 1.0e-12,
            "all_beta_in_bounds": bool(dataset["actual_bounds"].all()),
            "known_chart_only": bool(dataset["chart_id"].astype(str).str.len().gt(0).all()),
        },
        semantics="frozen_attempts_volume_stratified_quota_replay",
        source_attempts_sha256=manifest["source_artifacts"]["dense_attempts"]["sha256"],
        requested_rows=requested,
        written_rows=len(dataset),
        density_cv=density_cv,
        fill_distance_p95_mm=fill_p95_mm,
        fill_distance_max_mm=fill_max_mm,
        parent_gap_p95_deg=float(np.percentile(gap, 95)),
        parent_gap_max_deg=float(np.max(gap)),
        residual_p95_mm=float(np.percentile(dataset["residual_mm"], 95)),
        residual_max_mm=float(dataset["residual_mm"].max()),
        split_rows=dataset["split"].value_counts().sort_index().to_dict(),
        chart_rows=dataset["chart_id"].value_counts().sort_index().to_dict(),
    )
    summary_stage = output_root / STAGE_DIR["summary"]
    summary_stage.mkdir()
    summary_gate = _gate(
        summary_stage / "gate.json",
        {
            "source_primary_shell_gates_pass": bool(protocol_gate["gate_pass"]),
            "dense_quota_replay_pass": bool(dense_gate["gate_pass"]),
            "dataset_rows_match": len(dataset) == requested,
            "student_not_used_to_upgrade_shell": True,
            "automatic_chart_classifier_disabled": config["student"]["automatic_chart_classifier"] is False,
        },
        semantics="formal_primary_shell_dataset_dense_quota_replay",
        source_run_root=str(source_root),
        source_git_sha=manifest["source_git_sha"],
        replay_git_sha=git_sha,
        dataset_path=str(dataset_path),
        dataset_sha256=sha256_file(dataset_path),
        dataset_rows=len(dataset),
        student_training_status="not_started_by_primary_shell_pipeline",
    )
    stopped = None if summary_gate["gate_pass"] else "summary"
    report = {
        "protocol_id": PROTOCOL_ID,
        "preset": config["preset"],
        "output_root": str(output_root),
        "requested_stage": "dense_quota_replay",
        "executed_stages": ["protocol", "dense_dataset", "summary"],
        "stopped_after": stopped,
    }
    atomic_write_json(output_root / "run_report.json", report)
    return report


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    output_root = _output_root(config, project_root, args.output)
    if args.replay_dense_from:
        if not args.output:
            raise ValueError("--replay-dense-from requires an explicit --output")
        return run_dense_quota_replay(
            config,
            project_root,
            Path(args.replay_dense_from).resolve(),
            output_root,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    functions: Mapping[str, Callable[[], dict[str, Any]]] = {
        "protocol": lambda: stage_protocol(config, project_root, output_root),
        "shell_search": lambda: stage_shell_search(config, project_root, output_root),
        "surface_atlas": lambda: stage_surface_atlas(config, project_root, output_root),
        "radial_labels": lambda: stage_radial_labels(config, project_root, output_root),
        "dense_dataset": lambda: stage_dense_dataset(config, project_root, output_root),
        "summary": lambda: stage_summary(config, project_root, output_root),
    }
    requested = tuple(functions) if args.stage == "all" else (args.stage,)
    executed: list[str] = []
    stopped_after: str | None = None
    for stage_name in requested:
        gate_path = output_root / STAGE_DIR[stage_name] / "gate.json"
        if gate_path.is_file():
            report = json.loads(gate_path.read_text(encoding="utf-8"))
        else:
            index = tuple(functions).index(stage_name)
            for prerequisite in tuple(functions)[:index]:
                _require_gate(output_root, prerequisite)
            report = functions[stage_name]()
            executed.append(stage_name)
        if not bool(report.get("gate_pass")):
            stopped_after = stage_name
            break
    run_report = {
        "protocol_id": PROTOCOL_ID,
        "preset": args.preset,
        "output_root": str(output_root),
        "requested_stage": args.stage,
        "executed_stages": executed,
        "stopped_after": stopped_after,
    }
    atomic_write_json(output_root / "run_report.json", run_report)
    return run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(SOURCE_ROOT / "configs/bacra_v13_ellipsoidal_shell_atlas.yaml"),
    )
    parser.add_argument(
        "--preset", choices=("smoke", "pilot", "formal"), default="smoke"
    )
    parser.add_argument(
        "--project-root", default=str(project_root_from(SOURCE_ROOT))
    )
    parser.add_argument("--output")
    parser.add_argument("--replay-dense-from")
    parser.add_argument(
        "--stage", choices=("all", *(name for name, _ in STAGES)), default="all"
    )
    return parser


def main() -> int:
    report = run_pipeline(build_parser().parse_args())
    print(json.dumps(report, sort_keys=True, indent=2))
    return 2 if report["stopped_after"] is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
