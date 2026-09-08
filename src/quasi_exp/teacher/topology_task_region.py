"""Topology-preserving strict-Gold task regions for BACRA V12.2.

The module owns the complete transition from a finite capability sample pool
to an auditable task graph.  In particular, it never promotes every sample in
an occupied Gold voxel to Gold and never re-infers connectivity from a lossy
KNN down-sample.  Callers provide immutable capability/centerline frames and
receive the support graph, exact task nodes, typed edges, direction-specific
waypoints and a fail-closed report through one in-memory interface.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .canonical import weighted_damped_pinv
from .canonical_atlas import AtlasCandidate, AtlasTaskNode, ContinuationOutcome


XYZ_COLUMNS = ("x_m", "y_m", "z_m")
BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
CENTERLINE_XYZ_COLUMNS = ("target_x_m", "target_y_m", "target_z_m")
CENTERLINE_BETA_COLUMNS = tuple(
    f"teacher_beta{index}_rad" for index in range(1, 7)
)
_NEIGHBOURS_6 = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)
_POSITIVE_NEIGHBOURS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


@dataclass(frozen=True)
class TopologyTaskPolicy:
    """Frozen V12.2 support-graph, task-node and waypoint semantics."""

    voxel_mm: float = 7.5
    roi_mm: float = 10.0
    gold_margin_deg: float = 1.5
    waypoint_step_mm: float = 1.0
    difficult_quantile: float = 0.85
    expected_phase_count: int = 720
    expected_main_voxel_count: int | None = 2161
    expected_task_node_count: int | None = 2881
    expected_task_edge_count: int | None = 5546

    def __post_init__(self) -> None:
        for name in ("voxel_mm", "roi_mm", "gold_margin_deg", "waypoint_step_mm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < float(self.difficult_quantile) < 1.0:
            raise ValueError("difficult_quantile must lie in (0, 1)")
        if int(self.expected_phase_count) < 2:
            raise ValueError("expected_phase_count must be at least two")

    @property
    def fingerprint(self) -> str:
        return _canonical_sha(
            {
                "voxel_mm": self.voxel_mm,
                "roi_mm": self.roi_mm,
                "gold_margin_deg": self.gold_margin_deg,
                "waypoint_step_mm": self.waypoint_step_mm,
                "difficult_quantile": self.difficult_quantile,
                "expected_phase_count": self.expected_phase_count,
                "expected_main_voxel_count": self.expected_main_voxel_count,
                "expected_task_node_count": self.expected_task_node_count,
                "expected_task_edge_count": self.expected_task_edge_count,
            }
        )


@dataclass(frozen=True)
class TopologyTaskRegion:
    """Complete deterministic V12.2 task-region evidence."""

    support_voxels: pd.DataFrame
    strict_gold_seed_pool: pd.DataFrame
    task_nodes: pd.DataFrame
    task_edges: pd.DataFrame
    task_edge_waypoints: pd.DataFrame
    report: Mapping[str, Any]
    policy: TopologyTaskPolicy

    @property
    def candidate_admission_gate_pass(self) -> bool:
        return bool(self.report["candidate_admission_gate_pass"])


def _require_columns(frame: pd.DataFrame, names: Sequence[str], label: str) -> None:
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _voxel_keys(xyz_m: np.ndarray, voxel_mm: float) -> np.ndarray:
    return np.floor(
        np.asarray(xyz_m, dtype=float) / (float(voxel_mm) / 1000.0)
    ).astype(np.int64)


def _key(row: Sequence[int]) -> tuple[int, int, int]:
    values = np.asarray(row, dtype=np.int64).reshape(3)
    return int(values[0]), int(values[1]), int(values[2])


def _label_components(
    unique_keys: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[tuple[int, int, int], ...], ...]]:
    keys = tuple(sorted({_key(row) for row in unique_keys}))
    key_set = set(keys)
    label_by_key: dict[tuple[int, int, int], int] = {}
    components: list[tuple[tuple[int, int, int], ...]] = []
    for root in keys:
        if root in label_by_key:
            continue
        component_id = len(components)
        pending = [root]
        label_by_key[root] = component_id
        members: list[tuple[int, int, int]] = []
        while pending:
            current = pending.pop()
            members.append(current)
            for delta in _NEIGHBOURS_6:
                adjacent = tuple(current[index] + delta[index] for index in range(3))
                if adjacent in key_set and adjacent not in label_by_key:
                    label_by_key[adjacent] = component_id
                    pending.append(adjacent)
        components.append(tuple(sorted(members)))
    labels = np.asarray([label_by_key[_key(row)] for row in unique_keys], dtype=np.int64)
    return labels, tuple(components)


def _joint_margin_deg(beta_rad: np.ndarray, bounds_rad: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    bounds = np.asarray(bounds_rad, dtype=float).reshape(6, 2)
    return np.min(
        np.minimum(beta - bounds[:, 0], bounds[:, 1] - beta), axis=1
    ) * 180.0 / math.pi


def _graph_connected(node_count: int, edges: pd.DataFrame) -> bool:
    if node_count < 1:
        return False
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for row in edges.itertuples(index=False):
        left, right = int(row.left_node_id), int(row.right_node_id)
        if left < 0 or right < 0 or left >= node_count or right >= node_count:
            return False
        adjacency[left].append(right)
        adjacency[right].append(left)
    reached = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for other in adjacency[current]:
            if other not in reached:
                reached.add(other)
                pending.append(other)
    return len(reached) == node_count


def _waypoint_rows(
    edge_id: int,
    edge_type: str,
    left: np.ndarray,
    right: np.ndarray,
    step_m: float,
) -> list[dict[str, Any]]:
    distance = float(np.linalg.norm(np.asarray(right) - np.asarray(left)))
    segment_count = max(1, int(math.ceil(distance / float(step_m))))
    rows: list[dict[str, Any]] = []
    for direction, start, stop in (
        ("forward", np.asarray(left, float), np.asarray(right, float)),
        ("reverse", np.asarray(right, float), np.asarray(left, float)),
    ):
        for index in range(segment_count + 1):
            fraction = index / segment_count
            xyz = (1.0 - fraction) * start + fraction * stop
            rows.append(
                {
                    "edge_id": int(edge_id),
                    "edge_type": str(edge_type),
                    "direction": direction,
                    "waypoint_idx": int(index),
                    "waypoint_count": int(segment_count + 1),
                    "x_m": float(xyz[0]),
                    "y_m": float(xyz[1]),
                    "z_m": float(xyz[2]),
                }
            )
    return rows


def build_topology_task_region(
    capability_rows: pd.DataFrame,
    centerline_rows: pd.DataFrame,
    bounds_rad: np.ndarray,
    policy: TopologyTaskPolicy | None = None,
) -> TopologyTaskRegion:
    """Build a strict-Gold support graph and exact task graph.

    Gold is evaluated per sample before voxelisation.  The main component is
    ranked by centerline phase support, voxel count and stable component ID.
    Every main-component voxel is retained, so the task graph cannot lose a
    thin corridor through down-sampling.
    """

    active = policy or TopologyTaskPolicy()
    capability_required = (
        "sample_id",
        *XYZ_COLUMNS,
        *BETA_COLUMNS,
        "minimum_margin_deg",
        "kappa",
    )
    centerline_required = (
        "phase_idx",
        *CENTERLINE_XYZ_COLUMNS,
        *CENTERLINE_BETA_COLUMNS,
        "fk_residual_mm",
    )
    _require_columns(capability_rows, capability_required, "capability_rows")
    _require_columns(centerline_rows, centerline_required, "centerline_rows")
    bounds = np.asarray(bounds_rad, dtype=float)
    if (
        bounds.shape != (6, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("bounds_rad must be finite ordered shape (6, 2)")

    capability = capability_rows[list(capability_required)].copy()
    centerline = centerline_rows[list(centerline_required)].copy()
    capability = capability.sort_values("sample_id", kind="stable").reset_index(drop=True)
    centerline = centerline.sort_values("phase_idx", kind="stable").reset_index(drop=True)
    if centerline["phase_idx"].tolist() != list(range(len(centerline))):
        raise ValueError("centerline phases must be complete stable 0..N-1")
    if len(centerline) != int(active.expected_phase_count):
        raise ValueError(
            f"centerline phase count {len(centerline)} != {active.expected_phase_count}"
        )
    numeric = capability[
        [*XYZ_COLUMNS, *BETA_COLUMNS, "minimum_margin_deg", "kappa"]
    ].to_numpy(dtype=float)
    if not len(capability) or not np.isfinite(numeric).all():
        raise ValueError("capability_rows must be non-empty and finite")

    xyz = capability[list(XYZ_COLUMNS)].to_numpy(dtype=float)
    center_xyz = centerline[list(CENTERLINE_XYZ_COLUMNS)].to_numpy(dtype=float)
    distance_to_centerline = cKDTree(center_xyz).query(xyz, k=1)[0]
    strict = (
        capability["minimum_margin_deg"].to_numpy(dtype=float)
        >= float(active.gold_margin_deg) - 1.0e-12
    ) & (distance_to_centerline <= float(active.roi_mm) / 1000.0 + 1.0e-12)
    strict_rows = capability.loc[strict].copy()
    if strict_rows.empty:
        raise ValueError("no strict-Gold samples lie inside the registered ROI")
    strict_xyz = strict_rows[list(XYZ_COLUMNS)].to_numpy(dtype=float)
    strict_keys = _voxel_keys(strict_xyz, active.voxel_mm)
    strict_rows[["voxel_x", "voxel_y", "voxel_z"]] = strict_keys

    unique_keys = np.unique(strict_keys, axis=0)
    _labels, components = _label_components(unique_keys)
    key_to_component = {
        key: component_id
        for component_id, members in enumerate(components)
        for key in members
    }
    strict_rows["component_id"] = [
        key_to_component[_key(row)] for row in strict_keys
    ]
    component_reports: list[dict[str, Any]] = []
    for component_id, members in enumerate(components):
        component_rows = strict_rows[strict_rows["component_id"].eq(component_id)]
        distances = cKDTree(
            component_rows[list(XYZ_COLUMNS)].to_numpy(dtype=float)
        ).query(center_xyz, k=1)[0]
        phase_count = int(
            np.count_nonzero(distances <= float(active.roi_mm) / 1000.0 + 1.0e-12)
        )
        component_reports.append(
            {
                "component_id": int(component_id),
                "voxel_count": int(len(members)),
                "sample_count": int(len(component_rows)),
                "centerline_phase_count": phase_count,
                "centerline_support": float(phase_count / len(centerline)),
                "centerline_distance_p95_mm": float(np.percentile(distances, 95) * 1000.0),
                "centerline_distance_max_mm": float(np.max(distances) * 1000.0),
            }
        )
    ranked = sorted(
        component_reports,
        key=lambda row: (
            -row["centerline_phase_count"],
            -row["voxel_count"],
            row["component_id"],
        ),
    )
    main_report = ranked[0]
    main_id = int(main_report["component_id"])
    seed_pool = strict_rows[strict_rows["component_id"].eq(main_id)].copy()
    seed_pool = seed_pool.sort_values("sample_id", kind="stable").reset_index(drop=True)
    main_keys = tuple(
        sorted({_key(row) for row in seed_pool[["voxel_x", "voxel_y", "voxel_z"]].to_numpy()})
    )
    main_key_set = set(main_keys)

    voxel_m = float(active.voxel_mm) / 1000.0
    seed_xyz = seed_pool[list(XYZ_COLUMNS)].to_numpy(dtype=float)
    seed_keys = seed_pool[["voxel_x", "voxel_y", "voxel_z"]].to_numpy(dtype=np.int64)
    centers = (seed_keys.astype(float) + 0.5) * voxel_m
    seed_pool["distance_to_voxel_center_m"] = np.linalg.norm(seed_xyz - centers, axis=1)
    representative = (
        seed_pool.sort_values(
            [
                "voxel_x",
                "voxel_y",
                "voxel_z",
                "distance_to_voxel_center_m",
                "minimum_margin_deg",
                "kappa",
                "sample_id",
            ],
            ascending=[True, True, True, True, False, True, True],
            kind="stable",
        )
        .drop_duplicates(["voxel_x", "voxel_y", "voxel_z"], keep="first")
        .sort_values(["voxel_x", "voxel_y", "voxel_z"], kind="stable")
        .reset_index(drop=True)
    )
    if len(representative) != len(main_keys):
        raise RuntimeError("strict-Gold voxel witness selection is incomplete")

    boundary = []
    for key in main_keys:
        boundary.append(
            any(
                tuple(key[index] + delta[index] for index in range(3))
                not in main_key_set
                for delta in _NEIGHBOURS_6
            )
        )
    finite_kappa = representative["kappa"].to_numpy(dtype=float)
    difficult_threshold = float(np.quantile(finite_kappa, active.difficult_quantile))
    difficult = finite_kappa >= difficult_threshold

    support_nodes = pd.DataFrame(
        {
            "task_node_id": np.arange(len(representative), dtype=np.int64),
            "x_m": representative["x_m"].to_numpy(dtype=float),
            "y_m": representative["y_m"].to_numpy(dtype=float),
            "z_m": representative["z_m"].to_numpy(dtype=float),
            "minimum_margin_deg": representative["minimum_margin_deg"].to_numpy(dtype=float),
            "kappa": representative["kappa"].to_numpy(dtype=float),
            "voxel_x": representative["voxel_x"].to_numpy(dtype=np.int64),
            "voxel_y": representative["voxel_y"].to_numpy(dtype=np.int64),
            "voxel_z": representative["voxel_z"].to_numpy(dtype=np.int64),
            "node_role": "support_voxel",
            "is_boundary": np.asarray(boundary, dtype=bool),
            "is_difficult": difficult,
            "source_sample_id": representative["sample_id"].to_numpy(dtype=np.int64),
            "source_phase_idx": -1,
            "strict_gold": True,
        }
    )
    support_nodes["task_category"] = np.where(
        difficult,
        "difficult",
        np.where(np.asarray(boundary, dtype=bool), "boundary", "interior"),
    )
    for name in BETA_COLUMNS:
        support_nodes[name] = representative[name].to_numpy(dtype=float)

    center_beta = centerline[list(CENTERLINE_BETA_COLUMNS)].to_numpy(dtype=float)
    center_margin = _joint_margin_deg(center_beta, bounds)
    support_xyz = support_nodes[list(XYZ_COLUMNS)].to_numpy(dtype=float)
    attach_distance, attach_index = cKDTree(support_xyz).query(center_xyz, k=1)
    attached_kappa = support_nodes.iloc[np.asarray(attach_index, dtype=int)][
        "kappa"
    ].to_numpy(dtype=float)
    center_nodes = pd.DataFrame(
        {
            "task_node_id": np.arange(
                len(support_nodes), len(support_nodes) + len(centerline), dtype=np.int64
            ),
            "x_m": centerline["target_x_m"].to_numpy(dtype=float),
            "y_m": centerline["target_y_m"].to_numpy(dtype=float),
            "z_m": centerline["target_z_m"].to_numpy(dtype=float),
            "minimum_margin_deg": center_margin,
            "kappa": attached_kappa,
            "voxel_x": _voxel_keys(center_xyz, active.voxel_mm)[:, 0],
            "voxel_y": _voxel_keys(center_xyz, active.voxel_mm)[:, 1],
            "voxel_z": _voxel_keys(center_xyz, active.voxel_mm)[:, 2],
            "node_role": "centerline",
            "is_boundary": False,
            "is_difficult": False,
            "source_sample_id": -1,
            "source_phase_idx": centerline["phase_idx"].to_numpy(dtype=np.int64),
            "strict_gold": center_margin >= float(active.gold_margin_deg) - 1.0e-12,
            "task_category": "centerline",
        }
    )
    for index, name in enumerate(BETA_COLUMNS):
        center_nodes[name] = center_beta[:, index]
    center_nodes["fk_residual_mm"] = centerline["fk_residual_mm"].to_numpy(dtype=float)
    support_nodes["fk_residual_mm"] = 0.0
    task_nodes = pd.concat([support_nodes, center_nodes], ignore_index=True)

    node_by_key = {
        _key(row): int(node_id)
        for row, node_id in zip(
            support_nodes[["voxel_x", "voxel_y", "voxel_z"]].to_numpy(),
            support_nodes["task_node_id"],
        )
    }
    edge_rows: list[dict[str, Any]] = []

    def append_edge(left: int, right: int, edge_type: str) -> None:
        pair = (min(int(left), int(right)), max(int(left), int(right)))
        left_xyz = task_nodes.loc[pair[0], list(XYZ_COLUMNS)].to_numpy(dtype=float)
        right_xyz = task_nodes.loc[pair[1], list(XYZ_COLUMNS)].to_numpy(dtype=float)
        edge_rows.append(
            {
                "edge_id": len(edge_rows),
                "left_node_id": pair[0],
                "right_node_id": pair[1],
                "edge_type": edge_type,
                "length_m": float(np.linalg.norm(right_xyz - left_xyz)),
            }
        )

    for key in main_keys:
        left = node_by_key[key]
        for delta in _POSITIVE_NEIGHBOURS:
            adjacent = tuple(key[index] + delta[index] for index in range(3))
            if adjacent in node_by_key:
                append_edge(left, node_by_key[adjacent], "support_face")

    center_offset = len(support_nodes)
    for phase in range(len(centerline)):
        append_edge(
            center_offset + phase,
            center_offset + ((phase + 1) % len(centerline)),
            "centerline_cycle",
        )
    for phase, support_id in enumerate(np.asarray(attach_index, dtype=int)):
        append_edge(center_offset + phase, support_id, "centerline_attachment")

    task_edges = pd.DataFrame(edge_rows).sort_values("edge_id", kind="stable").reset_index(drop=True)
    duplicated_pairs = task_edges.duplicated(["left_node_id", "right_node_id"]).any()
    waypoint_rows: list[dict[str, Any]] = []
    for row in task_edges.itertuples(index=False):
        left_xyz = task_nodes.loc[int(row.left_node_id), list(XYZ_COLUMNS)].to_numpy(dtype=float)
        right_xyz = task_nodes.loc[int(row.right_node_id), list(XYZ_COLUMNS)].to_numpy(dtype=float)
        waypoint_rows.extend(
            _waypoint_rows(
                int(row.edge_id),
                str(row.edge_type),
                left_xyz,
                right_xyz,
                float(active.waypoint_step_mm) / 1000.0,
            )
        )
    waypoints = pd.DataFrame(waypoint_rows)
    waypoint_step = []
    for (_edge, _direction), group in waypoints.groupby(
        ["edge_id", "direction"], sort=True
    ):
        values = group.sort_values("waypoint_idx", kind="stable")[
            list(XYZ_COLUMNS)
        ].to_numpy(dtype=float)
        if len(values) > 1:
            waypoint_step.extend(np.linalg.norm(np.diff(values, axis=0), axis=1))
    max_waypoint_step_mm = (
        float(np.max(waypoint_step) * 1000.0) if waypoint_step else 0.0
    )
    graph_connected = _graph_connected(len(task_nodes), task_edges)
    checks = {
        "strict_gold_source_nonempty": bool(len(strict_rows)),
        "main_component_covers_centerline": int(main_report["centerline_phase_count"])
        == len(centerline),
        "main_component_voxel_count": active.expected_main_voxel_count is None
        or len(support_nodes) == int(active.expected_main_voxel_count),
        "support_witnesses_actual_gold": bool(
            support_nodes["minimum_margin_deg"].ge(active.gold_margin_deg - 1.0e-12).all()
        ),
        "centerline_complete": len(center_nodes) == int(active.expected_phase_count),
        "centerline_actual_gold": bool(center_nodes["strict_gold"].all()),
        "centerline_fk_residual": bool(
            center_nodes["fk_residual_mm"].le(3.0 + 1.0e-12).all()
        ),
        "task_node_count": active.expected_task_node_count is None
        or len(task_nodes) == int(active.expected_task_node_count),
        "task_edge_count": active.expected_task_edge_count is None
        or len(task_edges) == int(active.expected_task_edge_count),
        "task_graph_connected": graph_connected,
        "no_duplicate_task_edges": not bool(duplicated_pairs),
        "attachments_within_roi": bool(
            np.max(attach_distance) <= active.roi_mm / 1000.0 + 1.0e-12
        ),
        "waypoints_complete": int(waypoints["edge_id"].nunique()) == len(task_edges),
        "waypoint_step": max_waypoint_step_mm
        <= float(active.waypoint_step_mm) + 1.0e-9,
    }
    projected_checks = {
        name: checks[name]
        for name in (
            "strict_gold_source_nonempty",
            "main_component_covers_centerline",
            "main_component_voxel_count",
        )
    }
    discretization_checks = {
        name: value
        for name, value in checks.items()
        if name not in projected_checks
    }
    projected_pass = bool(all(projected_checks.values()))
    discretization_pass = bool(all(discretization_checks.values()))
    report = _json_safe(
        {
            "schema_version": 1,
            "gate_semantics": "topology_preserving_candidate_admission",
            "claim_scope": "simulation_canonical_atlas_and_student_diagnostics",
            "policy_fingerprint": active.fingerprint,
            "policy": {
                "voxel_mm": active.voxel_mm,
                "roi_mm": active.roi_mm,
                "adjacency": 6,
                "gold_margin_deg": active.gold_margin_deg,
                "waypoint_step_mm": active.waypoint_step_mm,
            },
            "component_reports": component_reports,
            "selected_main_component_id": main_id,
            "strict_gold_roi_sample_count": len(strict_rows),
            "main_component_seed_count": len(seed_pool),
            "main_component_voxel_count": len(support_nodes),
            "centerline_node_count": len(center_nodes),
            "task_node_count": len(task_nodes),
            "task_edge_count": len(task_edges),
            "task_edge_type_counts": task_edges["edge_type"].value_counts().to_dict(),
            "centerline_attachment_distance_p95_mm": float(
                np.percentile(attach_distance, 95) * 1000.0
            ),
            "centerline_attachment_distance_max_mm": float(
                np.max(attach_distance) * 1000.0
            ),
            "max_waypoint_step_mm": max_waypoint_step_mm,
            "projected_capability_checks": projected_checks,
            "task_discretization_checks": discretization_checks,
            "checks": checks,
            "projected_capability_gate_pass": projected_pass,
            "task_discretization_gate_pass": discretization_pass,
            "candidate_admission_gate_pass": projected_pass and discretization_pass,
            "gate_pass": projected_pass and discretization_pass,
            "deployment_claim_gate_pass": False,
            "evidence_limitations": [
                "7.5 mm voxel connectivity is finite task-space projection evidence, not proof that the strict-Gold beta set is connected",
                "task-edge waypoint paths require branch-lifted predictor-corrector validation before dataset admission",
                "the preregistered 5 mm / 6-neighbour sensitivity remains disconnected",
            ],
        }
    )
    seed_pool = seed_pool.drop(columns=["distance_to_voxel_center_m"], errors="ignore")
    support_voxels = support_nodes.copy()
    return TopologyTaskRegion(
        support_voxels=support_voxels,
        strict_gold_seed_pool=seed_pool,
        task_nodes=task_nodes,
        task_edges=task_edges,
        task_edge_waypoints=waypoints,
        report=MappingProxyType(dict(report)),
        policy=active,
    )


def waypoint_map_from_frame(
    edges: pd.DataFrame, waypoints: pd.DataFrame
) -> Mapping[tuple[int, int], np.ndarray]:
    """Return directed waypoint targets keyed by task-node direction."""

    _require_columns(
        edges,
        ("edge_id", "left_node_id", "right_node_id"),
        "edges",
    )
    _require_columns(
        waypoints,
        ("edge_id", "direction", "waypoint_idx", *XYZ_COLUMNS),
        "waypoints",
    )
    output: dict[tuple[int, int], np.ndarray] = {}
    edge_by_id = {
        int(row.edge_id): (int(row.left_node_id), int(row.right_node_id))
        for row in edges.itertuples(index=False)
    }
    for (edge_id, direction), group in waypoints.groupby(
        ["edge_id", "direction"], sort=True
    ):
        left, right = edge_by_id[int(edge_id)]
        key = (left, right) if str(direction) == "forward" else (right, left)
        values = group.sort_values("waypoint_idx", kind="stable")[
            list(XYZ_COLUMNS)
        ].to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"edge {edge_id} direction {direction} has invalid waypoints")
        output[key] = values[1:].copy()
    expected = 2 * len(edge_by_id)
    if len(output) != expected:
        raise ValueError(f"waypoint map has {len(output)} directions, expected {expected}")
    return MappingProxyType(output)


def make_strict_gold_witnessed_continuation(
    environment: Any,
    waypoint_map: Mapping[tuple[int, int], np.ndarray],
    *,
    damping: float,
    beta_weights: Sequence[float],
    max_corrector_iterations: int,
    residual_max_mm: float,
    gold_margin_deg: float,
) -> Any:
    """Create an edge-aware continuation adapter with all-waypoint Gold checks."""

    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    weights = np.asarray(beta_weights, dtype=float).reshape(6)

    def fk_one(beta: np.ndarray) -> np.ndarray:
        return np.asarray(
            environment.fk(np.asarray(beta, dtype=float).reshape(1, 6)),
            dtype=float,
        ).reshape(-1, 3)[0]

    def jacobian(beta: np.ndarray) -> np.ndarray:
        method = getattr(environment, "jacobian", None) or getattr(
            environment, "numerical_jacobian", None
        )
        if method is None:
            raise TypeError("environment lacks a Jacobian method")
        return np.asarray(method(np.asarray(beta, dtype=float)), dtype=float).reshape(3, 6)

    def continuation(
        source: AtlasCandidate, target: AtlasTaskNode
    ) -> ContinuationOutcome:
        path = waypoint_map.get((int(source.node_id), int(target.node_id)))
        if path is None:
            return ContinuationOutcome(
                beta_rad=source.beta_rad,
                residual_mm=1.0e300,
                success=False,
                actual_bounds=False,
                corrector_iterations=0,
                status="missing_waypoint_path",
                minimum_margin_deg=-1.0e300,
                waypoint_count=0,
            )
        beta = np.asarray(source.beta_rad, dtype=float).reshape(6).copy()
        total_nfev = 0
        min_margin = float(source.min_margin_deg)
        residual_mm = float(source.residual_mm)
        try:
            for waypoint_index, waypoint in enumerate(np.asarray(path, dtype=float)):
                current_xyz = fk_one(beta)
                predictor = beta + weighted_damped_pinv(
                    jacobian(beta), damping=float(damping), weights=weights
                ) @ (waypoint - current_xyz)
                predictor_in_bounds = bool(
                    np.all(predictor >= bounds[:, 0] - 1.0e-12)
                    and np.all(predictor <= bounds[:, 1] + 1.0e-12)
                )
                initial = predictor if predictor_in_bounds else beta

                def residual(values: np.ndarray) -> np.ndarray:
                    return (fk_one(values) - waypoint) / 0.001

                solved = least_squares(
                    residual,
                    initial,
                    bounds=(bounds[:, 0], bounds[:, 1]),
                    max_nfev=int(max_corrector_iterations),
                    xtol=1.0e-12,
                    ftol=1.0e-12,
                    gtol=1.0e-12,
                )
                beta = np.asarray(solved.x, dtype=float).reshape(6)
                total_nfev += int(solved.nfev)
                achieved = fk_one(beta)
                residual_mm = float(np.linalg.norm(achieved - waypoint) * 1000.0)
                actual_bounds = bool(
                    np.all(beta >= bounds[:, 0] - 1.0e-12)
                    and np.all(beta <= bounds[:, 1] + 1.0e-12)
                )
                margin = float(_joint_margin_deg(beta.reshape(1, 6), bounds)[0])
                min_margin = min(min_margin, margin)
                if not (
                    bool(solved.success)
                    and actual_bounds
                    and residual_mm <= float(residual_max_mm) + 1.0e-12
                    and margin >= float(gold_margin_deg) - 1.0e-12
                ):
                    return ContinuationOutcome(
                        beta_rad=beta,
                        residual_mm=residual_mm,
                        success=False,
                        actual_bounds=actual_bounds,
                        corrector_iterations=total_nfev,
                        status=f"waypoint_{waypoint_index}_failed",
                        minimum_margin_deg=min_margin,
                        waypoint_count=len(path),
                    )
            return ContinuationOutcome(
                beta_rad=beta,
                residual_mm=residual_mm,
                success=True,
                actual_bounds=True,
                corrector_iterations=total_nfev,
                status="strict_gold_waypoint_path",
                minimum_margin_deg=min_margin,
                waypoint_count=len(path),
            )
        except Exception as error:
            return ContinuationOutcome(
                beta_rad=beta,
                residual_mm=1.0e300,
                success=False,
                actual_bounds=False,
                corrector_iterations=total_nfev,
                status=f"exception:{type(error).__name__}",
                minimum_margin_deg=min_margin,
                waypoint_count=len(path),
            )

    return continuation
