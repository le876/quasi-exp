"""Branch-agnostic capability and task-region construction for BACRA-V12.

This module deliberately owns the complete capability-to-task-graph transition.
Callers provide an authoritative forward environment, a centerline and a small
policy; they receive immutable reports and on-disk evidence instead of having
to reproduce Sobol sampling, margin accounting, voxelisation, component
selection or graph construction themselves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc

from quasi_exp.io import load_config


class CapabilityEnvironment(Protocol):
    """The authoritative float64 FK/Jacobian seam used by the capability map."""

    bounds: np.ndarray

    def fk(self, beta: np.ndarray) -> np.ndarray: ...

    def jacobian(self, beta: np.ndarray) -> np.ndarray: ...


_NEIGHBOURS_6 = np.asarray(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    """Encode unavailable numeric diagnostics as JSON null, never NaN/Infinity."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _finite_array(value: np.ndarray, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}, got {array.shape}")
    return array


def load_beta_bounds_from_config(config_path: str | Path) -> np.ndarray:
    """Read the six authoritative beta bounds from the registered robot config."""

    path = Path(config_path)
    config = load_config(path)
    try:
        ranges = config["sampling"]["beta_ranges_rad"]
        bounds = np.asarray([ranges[f"beta{index}"] for index in range(1, 7)], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("robot config must define sampling.beta_ranges_rad beta1..beta6") from exc
    if bounds.shape != (6, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("robot config beta bounds must be finite ordered (6, 2) values")
    return bounds


@dataclass(frozen=True)
class CapabilityPolicy:
    """Frozen numerical and task-region policy for one capability materialisation."""

    robot_config_path: str | Path
    sobol_power: int = 20
    seed: int = 20260730
    task_seed: int = 20260731
    voxel_size_m: float = 0.010
    chunk_rows: int = 8192
    jacobian_eps_rad: float = 1.0e-4
    core_margin_deg: float = 1.5
    roi_radii_mm_desc: tuple[float, ...] = (20.0, 10.0, 5.0, 2.0)
    centerline_core_support_min: float = 0.95
    task_node_count: int = 2000
    knn_k: int = 12
    max_edge_to_median_nn: float = 1.5
    interior_fraction: float = 0.60
    boundary_fraction: float = 0.25
    difficult_fraction: float = 0.15

    def __post_init__(self) -> None:
        power = int(self.sobol_power)
        if power < 1:
            raise ValueError("sobol_power must be at least one")
        if not np.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        if int(self.chunk_rows) < 1 or int(self.task_node_count) < 2 or int(self.knn_k) < 1:
            raise ValueError("chunk_rows, task_node_count and knn_k must be positive")
        if not np.isfinite(self.core_margin_deg) or self.core_margin_deg <= 0.0:
            raise ValueError("core_margin_deg must be finite and positive")
        radii = tuple(float(value) for value in self.roi_radii_mm_desc)
        if not radii or any(not np.isfinite(value) or value <= 0.0 for value in radii):
            raise ValueError("roi_radii_mm_desc must contain finite positive values")
        if tuple(sorted(radii, reverse=True)) != radii:
            raise ValueError("roi_radii_mm_desc must be supplied in descending order")
        fractions = np.asarray(
            [self.interior_fraction, self.boundary_fraction, self.difficult_fraction], dtype=float
        )
        if not np.isfinite(fractions).all() or np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
            raise ValueError("interior/boundary/difficult fractions must be non-negative and sum to one")
        if not 0.0 <= float(self.centerline_core_support_min) <= 1.0:
            raise ValueError("centerline_core_support_min must lie in [0, 1]")
        if not np.isfinite(self.max_edge_to_median_nn) or self.max_edge_to_median_nn <= 0.0:
            raise ValueError("max_edge_to_median_nn must be finite and positive")
        object.__setattr__(self, "robot_config_path", Path(self.robot_config_path))
        object.__setattr__(self, "sobol_power", power)
        object.__setattr__(self, "roi_radii_mm_desc", radii)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], *, project_root: str | Path = ".") -> "CapabilityPolicy":
        """Adapt the V12 YAML sections without exposing runner policy plumbing."""

        root = Path(project_root)
        capability = dict(config["capability"])
        task = dict(config["task_region"])
        seeds = dict(config["seeds"])
        robot = root / str(config["robot_config"])
        rows = int(capability["pool_rows"])
        if rows <= 0 or rows & (rows - 1):
            raise ValueError("capability.pool_rows must be a power of two for nested Sobol sampling")
        return cls(
            robot_config_path=robot,
            sobol_power=int(np.log2(rows)),
            seed=int(seeds["capability"]),
            task_seed=int(seeds["task_region"]),
            voxel_size_m=float(capability["voxel_mm"]) / 1000.0,
            chunk_rows=int(capability["chunk_rows"]),
            jacobian_eps_rad=float(capability.get("jacobian_eps_rad", 1.0e-4)),
            roi_radii_mm_desc=tuple(float(value) for value in capability["roi_radii_mm_desc"]),
            centerline_core_support_min=float(capability["centerline_core_support_min"]),
            task_node_count=int(task["node_count"]),
            knn_k=int(task["knn_k"]),
            max_edge_to_median_nn=float(task["max_edge_to_median_nn"]),
            interior_fraction=float(task.get("interior_fraction", 0.60)),
            boundary_fraction=float(task.get("boundary_fraction", 0.25)),
            difficult_fraction=float(task.get("difficult_fraction", 0.15)),
        )

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["robot_config_path"] = str(self.robot_config_path)
        return _json_fingerprint(payload)


@dataclass(frozen=True)
class CapabilityGateReport:
    config_bounds_match: bool
    finite_fk_rate: float
    centerline_core_support: float
    connected_component_exists: bool
    task_graph_connected: bool
    enough_task_nodes: bool
    selected_roi_radius_mm: float | None
    gate_pass: bool
    checks: Mapping[str, bool]
    metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "schema_version": 1,
            "gate_semantics": "capability_task_region_admission",
            "claim_scope": "simulation_canonical_atlas_and_student_diagnostics",
            "config_bounds_match": self.config_bounds_match,
            "finite_fk_rate": self.finite_fk_rate,
            "centerline_core_support": self.centerline_core_support,
            "connected_component_exists": self.connected_component_exists,
            "task_graph_connected": self.task_graph_connected,
            "enough_task_nodes": self.enough_task_nodes,
            "selected_roi_radius_mm": self.selected_roi_radius_mm,
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
            "gate_pass": self.gate_pass,
        })


@dataclass(frozen=True)
class TaskRegion:
    radius_mm: float | None
    component_id: int | None
    nodes: pd.DataFrame
    edges: pd.DataFrame
    connected: bool
    median_nearest_neighbor_m: float
    centerline_core_support: float


@dataclass(frozen=True)
class CapabilityRegionManifest:
    """One complete, replayable capability map and selected task-region evidence."""

    policy_fingerprint: str
    robot_config_sha256: str
    bounds_rad: np.ndarray
    capability_rows: pd.DataFrame
    voxels: pd.DataFrame
    task_region: TaskRegion
    gate: CapabilityGateReport
    output_dir: Path | None = None

    @property
    def task_region_gate_pass(self) -> bool:
        return self.gate.gate_pass


def nested_sobol_beta(bounds_rad: np.ndarray, *, power: int, seed: int) -> np.ndarray:
    """Generate a scrambled Sobol prefix; larger powers retain every lower prefix."""

    bounds = _finite_array(bounds_rad, (6, 2), name="bounds_rad")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds_rad must be ordered")
    exponent = int(power)
    if exponent < 1:
        raise ValueError("power must be at least one")
    unit = qmc.Sobol(d=6, scramble=True, seed=int(seed)).random_base2(exponent)
    return bounds[:, 0][None, :] + unit * (bounds[:, 1] - bounds[:, 0])[None, :]


def batch_fk(environment: CapabilityEnvironment, beta_rad: np.ndarray, *, chunk_rows: int) -> np.ndarray:
    """Chunk authoritative vector FK calls without changing their float64 semantics."""

    beta = np.asarray(beta_rad, dtype=float)
    if beta.ndim != 2 or beta.shape[1] != 6 or not np.isfinite(beta).all():
        raise ValueError("beta_rad must be finite with shape (N, 6)")
    if int(chunk_rows) < 1:
        raise ValueError("chunk_rows must be positive")
    output = np.empty((len(beta), 3), dtype=float)
    for start in range(0, len(beta), int(chunk_rows)):
        stop = min(start + int(chunk_rows), len(beta))
        chunk = np.asarray(environment.fk(beta[start:stop]), dtype=float)
        if chunk.shape != (stop - start, 3):
            raise ValueError("environment.fk must return shape (N, 3)")
        output[start:stop] = chunk
    return output


def batch_jacobian_metrics(environment: CapabilityEnvironment, beta_rad: np.ndarray) -> dict[str, np.ndarray]:
    """Compute scalar-equivalent numerical Jacobian summaries in a stable row order."""

    beta = np.asarray(beta_rad, dtype=float)
    if beta.ndim != 2 or beta.shape[1] != 6 or not np.isfinite(beta).all():
        raise ValueError("beta_rad must be finite with shape (N, 6)")
    values = np.empty((len(beta), 4), dtype=float)
    for index, row in enumerate(beta):
        jacobian = np.asarray(environment.jacobian(row), dtype=float)
        if jacobian.shape != (3, 6) or not np.isfinite(jacobian).all():
            raise ValueError("environment.jacobian must return finite shape (3, 6)")
        singular = np.linalg.svd(jacobian, compute_uv=False)
        values[index, :3] = singular
        values[index, 3] = np.inf if singular[2] <= 0.0 else singular[0] / singular[2]
    return {"sigma1_m": values[:, 0], "sigma2_m": values[:, 1], "sigma3_m": values[:, 2], "kappa": values[:, 3]}


def _joint_margins(beta_rad: np.ndarray, bounds_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = beta_rad - bounds_rad[:, 0][None, :]
    upper = bounds_rad[:, 1][None, :] - beta_rad
    margin_rad = np.minimum(lower, upper)
    half_span = (bounds_rad[:, 1] - bounds_rad[:, 0]) / 2.0
    normalized = margin_rad / half_span[None, :]
    return np.rad2deg(margin_rad), np.rad2deg(np.min(margin_rad, axis=1)), normalized


def _voxel_keys(xyz_m: np.ndarray, voxel_size_m: float) -> np.ndarray:
    return np.floor(np.asarray(xyz_m, dtype=float) / float(voxel_size_m)).astype(np.int64)


def _key_tuple(row: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(value) for value in np.asarray(row, dtype=np.int64).reshape(3))


def _connected_components(keys: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int, int], int]]:
    unique = sorted({_key_tuple(row) for row in keys})
    key_set = set(unique)
    component: dict[tuple[int, int, int], int] = {}
    current = 0
    for root in unique:
        if root in component:
            continue
        queue = [root]
        component[root] = current
        while queue:
            node = queue.pop()
            for delta in _NEIGHBOURS_6:
                adjacent = (node[0] + int(delta[0]), node[1] + int(delta[1]), node[2] + int(delta[2]))
                if adjacent in key_set and adjacent not in component:
                    component[adjacent] = current
                    queue.append(adjacent)
        current += 1
    return np.asarray([component[_key_tuple(row)] for row in keys], dtype=np.int64), component


def _voxel_frame(rows: pd.DataFrame, voxel_keys: np.ndarray, *, core_margin_deg: float) -> tuple[pd.DataFrame, np.ndarray]:
    key_strings = np.asarray([f"{key[0]},{key[1]},{key[2]}" for key in voxel_keys], dtype=object)
    work = pd.DataFrame(
        {
            "voxel_id": key_strings,
            "voxel_x": voxel_keys[:, 0],
            "voxel_y": voxel_keys[:, 1],
            "voxel_z": voxel_keys[:, 2],
            "minimum_margin_deg": rows["minimum_margin_deg"].to_numpy(dtype=float),
            "kappa": rows["kappa"].to_numpy(dtype=float),
        }
    )
    grouped = work.groupby("voxel_id", sort=True, as_index=False).agg(
        voxel_x=("voxel_x", "first"),
        voxel_y=("voxel_y", "first"),
        voxel_z=("voxel_z", "first"),
        sample_count=("minimum_margin_deg", "size"),
        best_minimum_margin_deg=("minimum_margin_deg", "max"),
        median_kappa=("kappa", "median"),
    )
    grouped["capability_tier"] = np.where(
        grouped["best_minimum_margin_deg"].to_numpy(dtype=float) >= float(core_margin_deg),
        "Core-safe",
        "Feasible-boundary",
    )
    tier_lookup = dict(zip(grouped["voxel_id"], grouped["capability_tier"]))
    row_tier = np.asarray([tier_lookup[value] for value in key_strings], dtype=object)
    return grouped, row_tier


def _nearest_distance(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(target) == 0:
        return np.full(len(source), np.inf), np.full(len(source), -1, dtype=np.int64)
    distance, index = cKDTree(target).query(source, k=1)
    return np.asarray(distance, dtype=float), np.asarray(index, dtype=np.int64)


def _stable_farthest_indices(points: np.ndarray, candidates: np.ndarray, count: int, *, seed: int) -> list[int]:
    if count <= 0 or len(candidates) == 0:
        return []
    ordered = np.asarray(sorted(int(value) for value in candidates), dtype=np.int64)
    target = min(int(count), len(ordered))
    selected = [int(ordered[int(seed) % len(ordered)])]
    chosen = {selected[0]}
    min_distance = np.linalg.norm(points[ordered] - points[selected[0]], axis=1)
    while len(selected) < target:
        available = np.asarray([value not in chosen for value in ordered], dtype=bool)
        best_distance = np.max(min_distance[available])
        # Ordered candidates make ties independent of NumPy/scipy implementation details.
        next_local = int(np.flatnonzero(available & np.isclose(min_distance, best_distance, rtol=0.0, atol=1e-15))[0])
        next_index = int(ordered[next_local])
        selected.append(next_index)
        chosen.add(next_index)
        min_distance = np.minimum(min_distance, np.linalg.norm(points[ordered] - points[next_index], axis=1))
    return selected


def _task_categories(points: np.ndarray, voxel_keys: np.ndarray, kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_set = {_key_tuple(row) for row in voxel_keys}
    boundary = np.asarray(
        [
            any(
                (key[0] + int(delta[0]), key[1] + int(delta[1]), key[2] + int(delta[2])) not in key_set
                for delta in _NEIGHBOURS_6
            )
            for key in voxel_keys
        ],
        dtype=bool,
    )
    finite = np.where(np.isfinite(kappa), kappa, np.inf)
    threshold = float(np.quantile(finite, 0.85)) if len(finite) else np.inf
    difficult = finite >= threshold
    interior = ~boundary
    # Degenerate small point clouds should still permit quotas to be filled deterministically.
    if not np.any(interior):
        interior = np.ones(len(points), dtype=bool)
    if not np.any(boundary):
        boundary = np.ones(len(points), dtype=bool)
    if not np.any(difficult):
        difficult = np.ones(len(points), dtype=bool)
    return interior, boundary, difficult


def _build_task_graph(points: np.ndarray, *, knn_k: int, max_factor: float) -> tuple[pd.DataFrame, bool, float]:
    count = len(points)
    if count < 2:
        return pd.DataFrame(columns=["left_node_id", "right_node_id", "length_m"]), False, float("inf")
    tree = cKDTree(points)
    nearest, _ = tree.query(points, k=2)
    median = float(np.median(np.asarray(nearest)[:, 1]))
    if not np.isfinite(median) or median <= 0.0:
        return pd.DataFrame(columns=["left_node_id", "right_node_id", "length_m"]), False, median
    neighbour_count = min(count, int(knn_k) + 1)
    distances, neighbours = tree.query(points, k=neighbour_count)
    edges: dict[tuple[int, int], float] = {}
    limit = float(max_factor) * median
    for left in range(count):
        for distance, right in zip(np.atleast_1d(distances[left])[1:], np.atleast_1d(neighbours[left])[1:]):
            right_id = int(right)
            if left == right_id or float(distance) > limit:
                continue
            pair = (min(left, right_id), max(left, right_id))
            edges[pair] = min(edges.get(pair, float("inf")), float(distance))
    frame = pd.DataFrame(
        [{"left_node_id": left, "right_node_id": right, "length_m": length} for (left, right), length in sorted(edges.items())]
    )
    adjacent = [[] for _ in range(count)]
    for left, right in edges:
        adjacent[left].append(right)
        adjacent[right].append(left)
    visited = {0}
    queue = [0]
    while queue:
        node = queue.pop()
        for other in adjacent[node]:
            if other not in visited:
                visited.add(other)
                queue.append(other)
    return frame, len(visited) == count, median


def _select_task_region(
    rows: pd.DataFrame,
    voxel_keys: np.ndarray,
    centerline_xyz_m: np.ndarray,
    policy: CapabilityPolicy,
) -> TaskRegion:
    core = rows["capability_tier"].to_numpy(dtype=object) == "Core-safe"
    core_indices = np.flatnonzero(core)
    core_xyz = rows.loc[core, ["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    nearest_distance, nearest_core_local = _nearest_distance(centerline_xyz_m, core_xyz)
    selected: TaskRegion | None = None
    fallback_support = float(np.mean(nearest_distance <= max(policy.roi_radii_mm_desc) / 1000.0)) if len(nearest_distance) else 0.0
    for radius_mm in policy.roi_radii_mm_desc:
        radius_m = radius_mm / 1000.0
        supported = nearest_distance <= radius_m
        support = float(np.mean(supported)) if len(supported) else 0.0
        in_roi = np.flatnonzero(core & (_nearest_distance(rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float), centerline_xyz_m)[0] <= radius_m))
        if len(in_roi) < policy.task_node_count or support < policy.centerline_core_support_min:
            continue
        roi_keys = voxel_keys[in_roi]
        component_ids, _ = _connected_components(roi_keys)
        # A component earns support only for centerline points whose nearest safe sample is in it.
        support_by_component: list[tuple[float, int]] = []
        for component_id in sorted(set(component_ids.tolist())):
            component_global = in_roi[component_ids == component_id]
            component_set = set(int(value) for value in component_global)
            nearest_global = np.where(nearest_core_local >= 0, core_indices[nearest_core_local], -1)
            comp_support = float(np.mean(supported & np.asarray([value in component_set for value in nearest_global])))
            support_by_component.append((comp_support, int(component_id)))
        component_support, chosen_component = max(support_by_component, key=lambda pair: (pair[0], -pair[1]))
        component_global = in_roi[component_ids == chosen_component]
        if component_support < policy.centerline_core_support_min or len(component_global) < policy.task_node_count:
            continue
        points = rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
        component_keys = voxel_keys[component_global]
        component_kappa = rows.loc[component_global, "kappa"].to_numpy(dtype=float)
        interior, boundary, difficult = _task_categories(points[component_global], component_keys, component_kappa)
        quotas = (
            int(round(policy.task_node_count * policy.interior_fraction)),
            int(round(policy.task_node_count * policy.boundary_fraction)),
        )
        quotas = (quotas[0], quotas[1], policy.task_node_count - quotas[0] - quotas[1])
        picked: list[int] = []
        for category, quota, offset in zip((interior, boundary, difficult), quotas, (0, 1, 2)):
            local = np.flatnonzero(category)
            candidates = component_global[local]
            picked.extend(_stable_farthest_indices(points, candidates, quota, seed=policy.task_seed + offset))
        # Categories overlap by design; use one stable fill pass after quota selection.
        deduplicated = list(dict.fromkeys(picked))
        if len(deduplicated) < policy.task_node_count:
            remaining = np.asarray([index for index in component_global if int(index) not in set(deduplicated)], dtype=np.int64)
            deduplicated.extend(
                _stable_farthest_indices(points, remaining, policy.task_node_count - len(deduplicated), seed=policy.task_seed + 3)
            )
        chosen = np.asarray(deduplicated[: policy.task_node_count], dtype=np.int64)
        node_frame = rows.iloc[chosen].copy().reset_index(drop=False).rename(columns={"index": "capability_sample_id"})
        node_frame.insert(0, "task_node_id", np.arange(len(node_frame), dtype=np.int64))
        node_frame["component_id"] = int(chosen_component)
        difficult_global = set(map(int, component_global[np.flatnonzero(difficult)]))
        boundary_global = set(map(int, component_global[np.flatnonzero(boundary)]))
        node_frame["task_category"] = [
            "difficult"
            if int(index) in difficult_global
            else "boundary"
            if int(index) in boundary_global
            else "interior"
            for index in chosen
        ]
        edges, connected, median = _build_task_graph(
            node_frame[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
            knn_k=policy.knn_k,
            max_factor=policy.max_edge_to_median_nn,
        )
        region = TaskRegion(
            radius_mm=float(radius_mm),
            component_id=int(chosen_component),
            nodes=node_frame,
            edges=edges,
            connected=connected,
            median_nearest_neighbor_m=median,
            centerline_core_support=component_support,
        )
        if connected:
            return region
        selected = region
    if selected is not None:
        return selected
    return TaskRegion(
        radius_mm=None,
        component_id=None,
        nodes=pd.DataFrame(),
        edges=pd.DataFrame(columns=["left_node_id", "right_node_id", "length_m"]),
        connected=False,
        median_nearest_neighbor_m=float("inf"),
        centerline_core_support=fallback_support,
    )


def centerline_component_diagnostics(
    capability_rows: pd.DataFrame,
    centerline_xyz_m: np.ndarray,
    *,
    roi_radii_mm_desc: Sequence[float],
) -> Mapping[str, Any]:
    """Report safe support and connected-component support for one anchor.

    This is evidence only and does not alter task-region selection.  It makes
    the distinction between "a nearby Gold sample exists" and "one connected
    Gold component covers the closed task" explicit.
    """

    required = {
        "x_m",
        "y_m",
        "z_m",
        "voxel_x",
        "voxel_y",
        "voxel_z",
        "capability_tier",
    }
    missing = sorted(required - set(capability_rows.columns))
    if missing:
        raise ValueError(f"capability rows missing columns: {missing}")
    centerline = np.asarray(centerline_xyz_m, dtype=float)
    if centerline.ndim != 2 or centerline.shape[1] != 3 or not np.isfinite(centerline).all():
        raise ValueError("centerline_xyz_m must have finite shape (N, 3)")
    xyz = capability_rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    core = capability_rows["capability_tier"].eq("Core-safe").to_numpy()
    core_indices = np.flatnonzero(core)
    core_xyz = xyz[core]
    centerline_distance, nearest_core_local = _nearest_distance(centerline, core_xyz)
    nearest_global = np.where(
        nearest_core_local >= 0, core_indices[nearest_core_local], -1
    )
    distance_to_centerline = _nearest_distance(xyz, centerline)[0]
    rows: list[dict[str, Any]] = []
    for radius_mm in roi_radii_mm_desc:
        radius_m = float(radius_mm) / 1000.0
        in_roi = np.flatnonzero(core & (distance_to_centerline <= radius_m))
        support = float(np.mean(centerline_distance <= radius_m))
        if len(in_roi):
            keys = capability_rows.iloc[in_roi][
                ["voxel_x", "voxel_y", "voxel_z"]
            ].to_numpy(dtype=np.int64)
            component_ids, _components = _connected_components(keys)
            component_rows = []
            for component_id in sorted(set(component_ids.tolist())):
                members = in_roi[component_ids == component_id]
                member_set = set(map(int, members))
                component_support = float(
                    np.mean(
                        (centerline_distance <= radius_m)
                        & np.asarray(
                            [int(value) in member_set for value in nearest_global]
                        )
                    )
                )
                component_rows.append((component_support, len(members), component_id))
            best_support, best_count, best_id = max(
                component_rows, key=lambda item: (item[0], item[1], -item[2])
            )
        else:
            component_rows = []
            best_support, best_count, best_id = 0.0, 0, None
        rows.append(
            {
                "radius_mm": float(radius_mm),
                "centerline_core_support": support,
                "core_sample_count": int(len(in_roi)),
                "component_count": int(len(component_rows)),
                "largest_component_sample_count": int(
                    max((value[1] for value in component_rows), default=0)
                ),
                "best_component_id": (
                    None if best_id is None else int(best_id)
                ),
                "best_component_sample_count": int(best_count),
                "best_component_centerline_support": float(best_support),
            }
        )
    return {
        "schema_version": 1,
        "centerline_point_count": int(len(centerline)),
        "core_capability_sample_count": int(np.count_nonzero(core)),
        "radii": rows,
    }


def _write_artifacts(manifest: CapabilityRegionManifest, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.capability_rows.to_parquet(output_dir / "capability_map.parquet", index=False)
    manifest.voxels.to_parquet(output_dir / "capability_voxels.parquet", index=False)
    manifest.task_region.nodes.to_parquet(output_dir / "task_nodes.parquet", index=False)
    manifest.task_region.edges.to_parquet(output_dir / "task_edges.parquet", index=False)
    (output_dir / "gate.json").write_text(json.dumps(manifest.gate.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "policy_fingerprint": manifest.policy_fingerprint,
        "robot_config_sha256": manifest.robot_config_sha256,
        "bounds_rad": manifest.bounds_rad.tolist(),
        "capability_row_count": int(len(manifest.capability_rows)),
        "voxel_count": int(len(manifest.voxels)),
        "task_node_count": int(len(manifest.task_region.nodes)),
        "task_edge_count": int(len(manifest.task_region.edges)),
        "selected_roi_radius_mm": manifest.task_region.radius_mm,
        "component_id": manifest.task_region.component_id,
        "gate": manifest.gate.to_dict(),
    }
    (output_dir / "capability_manifest.json").write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")


def materialize_capability_region(
    environment: CapabilityEnvironment,
    primary_centerline: np.ndarray,
    policy: CapabilityPolicy,
    output_dir: str | Path | None = None,
) -> CapabilityRegionManifest:
    """Materialise a V12 capability map and the largest admissible A2 task ROI.

    The function is deliberately fail-closed: an incompatible environment/config
    bounds pair raises immediately, while an unsupported or disconnected task
    region returns a manifest with ``gate_pass=False`` and complete evidence.
    """

    config_path = Path(policy.robot_config_path)
    bounds = load_beta_bounds_from_config(config_path)
    environment_bounds = np.asarray(environment.bounds, dtype=float)
    if environment_bounds.shape != (6, 2) or not np.array_equal(environment_bounds, bounds):
        raise ValueError("environment.bounds must exactly match robot config beta bounds")
    centerline = np.asarray(primary_centerline, dtype=float)
    if centerline.ndim != 2 or centerline.shape[1] != 3 or len(centerline) < 2 or not np.isfinite(centerline).all():
        raise ValueError("primary_centerline must be finite with shape (N, 3), N >= 2")
    beta = nested_sobol_beta(bounds, power=policy.sobol_power, seed=policy.seed)
    xyz = batch_fk(environment, beta, chunk_rows=policy.chunk_rows)
    metrics = batch_jacobian_metrics(environment, beta)
    return materialize_capability_region_from_samples(
        environment,
        centerline,
        policy,
        beta_rad=beta,
        xyz_m=xyz,
        jacobian_metrics=metrics,
        output_dir=output_dir,
    )


def materialize_capability_region_from_samples(
    environment: CapabilityEnvironment,
    primary_centerline: np.ndarray,
    policy: CapabilityPolicy,
    *,
    beta_rad: np.ndarray,
    xyz_m: np.ndarray,
    jacobian_metrics: Mapping[str, np.ndarray],
    output_dir: str | Path | None = None,
    require_complete_sobol_prefix: bool = True,
) -> CapabilityRegionManifest:
    """Finalize a capability region from deterministic precomputed shards.

    This is the parallel execution seam.  V12 callers retain the default
    fail-closed requirement that the pool is exactly the frozen Sobol prefix.
    A registered follow-up protocol may set
    ``require_complete_sobol_prefix=False`` for a deterministic augmented pool
    (for example Sobol plus preregistered local enrichment); voxel, ROI, graph
    and Gate semantics are otherwise identical and remain owned here.
    """

    config_path = Path(policy.robot_config_path)
    bounds = load_beta_bounds_from_config(config_path)
    environment_bounds = np.asarray(environment.bounds, dtype=float)
    if environment_bounds.shape != (6, 2) or not np.array_equal(environment_bounds, bounds):
        raise ValueError("environment.bounds must exactly match robot config beta bounds")
    centerline = np.asarray(primary_centerline, dtype=float)
    if centerline.ndim != 2 or centerline.shape[1] != 3 or len(centerline) < 2 or not np.isfinite(centerline).all():
        raise ValueError("primary_centerline must be finite with shape (N, 3), N >= 2")
    beta = np.asarray(beta_rad, dtype=float)
    xyz = np.asarray(xyz_m, dtype=float)
    if (
        beta.ndim != 2
        or beta.shape[1] != 6
        or len(beta) < 1
        or not np.isfinite(beta).all()
    ):
        raise ValueError("beta_rad must be a non-empty finite sample pool with shape (N, 6)")
    if require_complete_sobol_prefix and beta.shape != (
        2 ** int(policy.sobol_power),
        6,
    ):
        raise ValueError("beta_rad must be the complete finite frozen Sobol prefix")
    if xyz.shape != (len(beta), 3):
        raise ValueError("xyz_m must align with beta_rad")
    finite_fk = np.isfinite(xyz).all(axis=1)
    if not np.all(finite_fk):
        raise ValueError("authoritative FK produced non-finite capability samples")
    required_metrics = ("sigma1_m", "sigma2_m", "sigma3_m", "kappa")
    metrics = {
        name: np.asarray(jacobian_metrics[name], dtype=float).reshape(-1)
        for name in required_metrics
    }
    if any(len(values) != len(beta) for values in metrics.values()):
        raise ValueError("jacobian metrics must align with beta_rad")
    if any(np.isnan(values).any() for values in metrics.values()):
        raise ValueError("jacobian metrics must not contain NaN")
    joint_margin_deg, minimum_margin_deg, normalized_margin = _joint_margins(beta, bounds)
    voxel_keys = _voxel_keys(xyz, policy.voxel_size_m)
    rows: dict[str, Any] = {
        "sample_id": np.arange(len(beta), dtype=np.int64),
        "x_m": xyz[:, 0],
        "y_m": xyz[:, 1],
        "z_m": xyz[:, 2],
        "minimum_margin_deg": minimum_margin_deg,
        "normalized_minimum_margin": np.min(normalized_margin, axis=1),
        **metrics,
        "voxel_x": voxel_keys[:, 0],
        "voxel_y": voxel_keys[:, 1],
        "voxel_z": voxel_keys[:, 2],
    }
    for index in range(6):
        rows[f"beta{index + 1}_rad"] = beta[:, index]
        rows[f"joint{index + 1}_margin_deg"] = joint_margin_deg[:, index]
        rows[f"joint{index + 1}_normalized_margin"] = normalized_margin[:, index]
    if hasattr(environment, "theta"):
        theta = np.asarray(getattr(environment, "theta")(beta), dtype=float)
        if theta.ndim == 2 and theta.shape[0] == len(beta) and np.isfinite(theta).all():
            for index in range(theta.shape[1]):
                rows[f"theta{index + 1}_rad"] = theta[:, index]
    frame = pd.DataFrame(rows)
    voxels, row_tier = _voxel_frame(frame, voxel_keys, core_margin_deg=policy.core_margin_deg)
    frame["capability_tier"] = row_tier
    task_region = _select_task_region(frame, voxel_keys, centerline, policy)
    checks = {
        "config_bounds_match": True,
        "finite_fk": bool(np.all(finite_fk)),
        "centerline_core_support": task_region.centerline_core_support >= policy.centerline_core_support_min,
        "connected_component": task_region.component_id is not None,
        "enough_task_nodes": len(task_region.nodes) == policy.task_node_count,
        "task_graph_connected": task_region.connected,
    }
    gate = CapabilityGateReport(
        config_bounds_match=True,
        finite_fk_rate=float(np.mean(finite_fk)),
        centerline_core_support=float(task_region.centerline_core_support),
        connected_component_exists=task_region.component_id is not None,
        task_graph_connected=task_region.connected,
        enough_task_nodes=len(task_region.nodes) == policy.task_node_count,
        selected_roi_radius_mm=task_region.radius_mm,
        gate_pass=all(checks.values()),
        checks=checks,
        metrics={
            "capability_sample_count": float(len(frame)),
            "occupied_voxel_count": float(len(voxels)),
            "core_safe_voxel_count": float(np.count_nonzero(voxels["capability_tier"].eq("Core-safe"))),
            "task_edge_count": float(len(task_region.edges)),
            "median_nearest_neighbor_m": float(task_region.median_nearest_neighbor_m),
        },
    )
    target = Path(output_dir) if output_dir is not None else None
    manifest = CapabilityRegionManifest(
        policy_fingerprint=policy.fingerprint,
        robot_config_sha256=_sha256(config_path),
        bounds_rad=bounds.copy(),
        capability_rows=frame,
        voxels=voxels,
        task_region=task_region,
        gate=gate,
        output_dir=target,
    )
    if target is not None:
        _write_artifacts(manifest, target)
    return manifest
