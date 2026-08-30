"""Continuous annular filling and independent continuity audits for retry17.

The helpers in this module deliberately operate only on task-space targets and
already legal candidate labels.  They never use proposal beta as a label,
seed, warm start, or branch hint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .exploration_qualification import weighted_beta_rms_deg
from .retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS, stable_id
from .retry15_canonical_graph import legal_candidate_clusters, mutual_knn_edges, select_t1


@dataclass(frozen=True)
class ContinuityPolicy:
    phase_a_fill_mm: float = 25.0
    phase_b_fill_mm: float = 20.0
    minimum_separation_mm: float = 15.0
    batch_size: int = 1000
    geometric_k: int = 16
    maximum_edge_spacing_factor: float = 1.75
    direction_projection_mm: float = 5.0
    direction_cosine_minimum: float = 0.5

    def __post_init__(self) -> None:
        if min(self.phase_a_fill_mm, self.phase_b_fill_mm, self.minimum_separation_mm) <= 0:
            raise ValueError("continuity distances must be positive")
        if self.phase_b_fill_mm >= self.phase_a_fill_mm:
            raise ValueError("phase B must tighten the fill target")
        if self.batch_size < 1 or self.geometric_k < 1:
            raise ValueError("batch size and graph k must be positive")
        if not 0 < self.direction_cosine_minimum <= 1:
            raise ValueError("direction cosine must be in (0, 1]")


def _profile_arrays(profile: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"u_center_mm", "inner_radius_mm", "outer_radius_mm"}
    if not required <= set(profile):
        raise ValueError("annular profile is missing required columns")
    ordered = profile.sort_values("u_center_mm", kind="stable")
    u = ordered["u_center_mm"].to_numpy(float)
    inner = ordered["inner_radius_mm"].to_numpy(float)
    outer = ordered["outer_radius_mm"].to_numpy(float)
    if len(u) < 2 or not np.isfinite(np.column_stack([u, inner, outer])).all():
        raise ValueError("annular profile must contain at least two finite rows")
    if np.any(np.diff(u) <= 0) or np.any(inner < 0) or np.any(outer <= inner):
        raise ValueError("annular profile is not a positive ordered tube")
    return u, inner, outer


def interpolate_profile(profile: pd.DataFrame, u_mm: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    u, inner, outer = _profile_arrays(profile)
    query = np.asarray(u_mm, dtype=float)
    return np.interp(query, u, inner), np.interp(query, u, outer)


def annular_coordinates(xyz_m: np.ndarray, *, zero_x_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(xyz_m, dtype=float).reshape(-1, 3)
    u = 1000.0 * (float(zero_x_m) - points[:, 0])
    rho = 1000.0 * np.hypot(points[:, 1], points[:, 2])
    phi = np.arctan2(points[:, 2], points[:, 1])
    return u, rho, phi


def points_inside_profile(
    xyz_m: np.ndarray,
    profile: pd.DataFrame,
    *,
    zero_x_m: float,
    erosion_mm: float = 0.0,
) -> np.ndarray:
    u_axis, _inner_axis, _outer_axis = _profile_arrays(profile)
    u, rho, _phi = annular_coordinates(xyz_m, zero_x_m=zero_x_m)
    inner, outer = interpolate_profile(profile, np.clip(u, u_axis[0], u_axis[-1]))
    return (
        (u >= u_axis[0] + erosion_mm)
        & (u <= u_axis[-1] - erosion_mm)
        & (rho >= inner + erosion_mm)
        & (rho <= outer - erosion_mm)
    )


def continuous_annular_sobol(
    profile: pd.DataFrame,
    *,
    power: int,
    seed: int,
    zero_x_m: float,
    pool_id: str,
) -> pd.DataFrame:
    """Draw volume-correct quotient targets with continuous ``u, rho, phi``."""

    if power < 1:
        raise ValueError("Sobol power must be positive")
    u_axis, _inner, _outer = _profile_arrays(profile)
    sample = qmc.Sobol(d=3, scramble=True, seed=int(seed)).random_base2(int(power))
    u_mm = u_axis[0] + sample[:, 0] * (u_axis[-1] - u_axis[0])
    r_min, r_max = interpolate_profile(profile, u_mm)
    rho_mm = np.sqrt(np.square(r_min) + sample[:, 1] * (np.square(r_max) - np.square(r_min)))
    phi = 0.5 * np.pi * sample[:, 2]
    xyz = np.column_stack(
        [
            float(zero_x_m) - u_mm / 1000.0,
            rho_mm * np.cos(phi) / 1000.0,
            rho_mm * np.sin(phi) / 1000.0,
        ]
    )
    records = pd.DataFrame(xyz, columns=XYZ_COLUMNS)
    records["u_mm"] = u_mm
    records["rho_mm"] = rho_mm
    records["phi_rad"] = phi
    records["pool_ordinal"] = np.arange(len(records), dtype=np.int64)
    records["pool_id"] = str(pool_id)
    records["target_id"] = [stable_id("retry17_target", pool_id, int(i)) for i in records["pool_ordinal"]]
    return records


def progressive_farthest_fill(
    pool: pd.DataFrame,
    existing_xyz_m: np.ndarray,
    *,
    maximum_new_points: int,
    minimum_separation_mm: float,
    batch_size: int,
    target_fill_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily fill the largest current holes; return points and batch audit.

    This exact progressive update is intentionally independent of the
    evaluation probes.  Stable pool order breaks all equal-distance ties.
    """

    if maximum_new_points < 0 or batch_size < 1:
        raise ValueError("invalid progressive fill budget")
    points = pool.loc[:, XYZ_COLUMNS].to_numpy(float)
    existing = np.asarray(existing_xyz_m, dtype=float).reshape(-1, 3)
    if len(existing):
        distance_mm = cKDTree(existing).query(points, k=1)[0] * 1000.0
    else:
        distance_mm = np.full(len(points), math.inf)
    available = np.ones(len(points), dtype=bool)
    chosen: list[int] = []
    batches: list[dict[str, Any]] = []
    for ordinal in range(int(maximum_new_points)):
        masked = np.where(available, distance_mm, -math.inf)
        index = int(np.argmax(masked))
        farthest = float(masked[index])
        if not math.isfinite(farthest) or farthest < float(minimum_separation_mm) - 1.0e-9:
            break
        chosen.append(index)
        available[index] = False
        delta_mm = np.linalg.norm(points - points[index], axis=1) * 1000.0
        distance_mm = np.minimum(distance_mm, delta_mm)
        if (ordinal + 1) % int(batch_size) == 0 or ordinal + 1 == int(maximum_new_points):
            residual = distance_mm[available]
            batches.append(
                {
                    "batch_index": len(batches),
                    "selected_total": len(chosen),
                    "pool_fill_p95_mm": float(np.percentile(residual, 95)) if len(residual) else 0.0,
                    "pool_fill_maximum_mm": float(np.max(residual)) if len(residual) else 0.0,
                    "target_fill_reached": bool(len(residual) == 0 or np.percentile(residual, 95) <= target_fill_mm),
                }
            )
    selected = pool.iloc[chosen].copy().reset_index(drop=True)
    if len(selected):
        selected["selection_ordinal"] = np.arange(len(selected), dtype=np.int64)
        selected["selection_batch"] = selected["selection_ordinal"] // int(batch_size)
    return selected, pd.DataFrame(batches)


def coverage_distance(probes: pd.DataFrame, labels: pd.DataFrame) -> Mapping[str, float]:
    if probes.empty or labels.empty:
        return {"p50_mm": math.inf, "p90_mm": math.inf, "p95_mm": math.inf, "maximum_mm": math.inf}
    distance = cKDTree(labels.loc[:, XYZ_COLUMNS].to_numpy(float)).query(
        probes.loc[:, XYZ_COLUMNS].to_numpy(float), k=1
    )[0] * 1000.0
    return {
        "p50_mm": float(np.percentile(distance, 50)),
        "p90_mm": float(np.percentile(distance, 90)),
        "p95_mm": float(np.percentile(distance, 95)),
        "maximum_mm": float(np.max(distance)),
    }


def local_directional_coverage(
    targets: pd.DataFrame,
    profile: pd.DataFrame,
    *,
    zero_x_m: float,
    neighbourhood_radius_mm: float,
    projection_minimum_mm: float = 5.0,
    cosine_minimum: float = 0.5,
) -> tuple[float, pd.DataFrame]:
    """Audit six signed local cylindrical directions on eroded interior points."""

    xyz = targets.loc[:, XYZ_COLUMNS].to_numpy(float)
    u, rho, phi = annular_coordinates(xyz, zero_x_m=zero_x_m)
    interior = points_inside_profile(
        xyz, profile, zero_x_m=zero_x_m, erosion_mm=float(neighbourhood_radius_mm)
    )
    tree = cKDTree(xyz)
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(interior):
        neighbours = [j for j in tree.query_ball_point(xyz[index], neighbourhood_radius_mm / 1000.0) if j != index]
        support = np.zeros(6, dtype=bool)
        for other in neighbours:
            dphi = math.atan2(math.sin(phi[other] - phi[index]), math.cos(phi[other] - phi[index]))
            delta = np.asarray([u[other] - u[index], rho[other] - rho[index], rho[index] * dphi])
            norm = float(np.linalg.norm(delta))
            if norm <= 1.0e-12:
                continue
            for axis in range(3):
                for sign_index, sign in enumerate((-1.0, 1.0)):
                    projection = sign * delta[axis]
                    if projection >= projection_minimum_mm and projection / norm >= cosine_minimum:
                        support[2 * axis + sign_index] = True
        rows.append(
            {
                "target_id": str(targets.iloc[index]["target_id"]),
                "interior": True,
                "minus_u": support[0], "plus_u": support[1],
                "minus_rho": support[2], "plus_rho": support[3],
                "minus_phi": support[4], "plus_phi": support[5],
                "all_six_supported": bool(np.all(support)),
                "neighbour_count": len(neighbours),
            }
        )
    audit = pd.DataFrame(rows)
    return (float(audit["all_six_supported"].mean()) if len(audit) else 0.0), audit


def geometric_graph(targets: pd.DataFrame, *, k: int, maximum_distance_mm: float) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["left_target_id", "right_target_id"])
    return mutual_knn_edges(targets, empty, k=int(k), maximum_distance_mm=float(maximum_distance_mm))


def graph_lcc_fraction(targets: pd.DataFrame, edges: pd.DataFrame) -> float:
    if targets.empty:
        return 0.0
    ids = targets["target_id"].astype(str).tolist()
    position = {value: i for i, value in enumerate(ids)}
    rows, cols = [], []
    for edge in edges.to_dict("records"):
        left, right = position.get(str(edge["left_target_id"])), position.get(str(edge["right_target_id"]))
        if left is None or right is None:
            continue
        rows.extend([left, right]); cols.extend([right, left])
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(ids)))
    count, labels = connected_components(graph, directed=False)
    return float(np.bincount(labels, minlength=count).max() / len(ids))


def graph_stretch(
    targets: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    pair_count: int,
    seed: int,
) -> tuple[Mapping[str, float], pd.DataFrame]:
    ids = targets["target_id"].astype(str).tolist()
    xyz = targets.loc[:, XYZ_COLUMNS].to_numpy(float) * 1000.0
    position = {value: i for i, value in enumerate(ids)}
    rows, cols, data = [], [], []
    for edge in edges.to_dict("records"):
        left, right = position.get(str(edge["left_target_id"])), position.get(str(edge["right_target_id"]))
        if left is None or right is None:
            continue
        distance = float(edge["distance_mm"])
        rows.extend([left, right]); cols.extend([right, left]); data.extend([distance, distance])
    graph = csr_matrix((data, (rows, cols)), shape=(len(ids), len(ids)))
    rng = np.random.default_rng(int(seed))
    if len(ids) < 2:
        return {"p50": math.inf, "p95": math.inf, "maximum": math.inf}, pd.DataFrame()
    raw_pairs = rng.integers(0, len(ids), size=(max(pair_count * 2, 1), 2))
    pairs = raw_pairs[raw_pairs[:, 0] != raw_pairs[:, 1]][:pair_count]
    source_ids = np.unique(pairs[:, 0])
    distances = dijkstra(graph, directed=False, indices=source_ids)
    source_row = {int(source): row for row, source in enumerate(source_ids)}
    records = []
    for left, right in pairs:
        euclidean = float(np.linalg.norm(xyz[left] - xyz[right]))
        path = float(distances[source_row[int(left)], int(right)])
        stretch = path / euclidean if euclidean > 1.0e-12 else 1.0
        records.append({"left_target_id": ids[left], "right_target_id": ids[right], "euclidean_mm": euclidean, "graph_path_mm": path, "stretch": stretch})
    audit = pd.DataFrame(records)
    finite = audit.loc[np.isfinite(audit["stretch"]), "stretch"].to_numpy(float)
    metrics = {
        "p50": float(np.percentile(finite, 50)) if len(finite) else math.inf,
        "p95": float(np.percentile(finite, 95)) if len(finite) else math.inf,
        "maximum": float(np.max(finite)) if len(finite) else math.inf,
        "finite_fraction": float(len(finite) / len(audit)) if len(audit) else 0.0,
    }
    return metrics, audit


def density_normalized_select_t2(
    candidates: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    pairwise_lambda: float,
    sigma_mm: float,
    locked_target_ids: Sequence[str] = (),
    tau_deg: float = 7.0,
    weights: Sequence[float] = (4, 4, 2, 2, 1, 1),
    maximum_sweeps: int = 30,
) -> pd.DataFrame:
    """Deterministic ICM with density-normalized undirected pair coefficients."""

    if pairwise_lambda == 0:
        return select_t1(candidates, weights=weights).assign(teacher="T1_control", pairwise_lambda=0.0)
    legal = legal_candidate_clusters(candidates, weights=weights)
    grouped = {str(key): value.sort_values("candidate_id", kind="stable").reset_index(drop=True) for key, value in legal.groupby("target_id", sort=True)}
    if not grouped:
        return legal
    adjacency: dict[str, list[tuple[str, float]]] = {key: [] for key in grouped}
    raw_edges = []
    degree = {key: 0 for key in grouped}
    for row in edges.to_dict("records"):
        left, right = str(row["left_target_id"]), str(row["right_target_id"])
        if left in grouped and right in grouped and left != right:
            distance = float(row["distance_mm"])
            raw_edges.append((left, right, distance)); degree[left] += 1; degree[right] += 1
    for left, right, distance in raw_edges:
        coefficient = math.exp(-(distance * distance) / (2.0 * sigma_mm * sigma_mm)) * (1.0 / degree[left] + 1.0 / degree[right])
        adjacency[left].append((right, coefficient)); adjacency[right].append((left, coefficient))
    beta = {key: frame.loc[:, BETA_COLUMNS].to_numpy(float) for key, frame in grouped.items()}
    unary = {
        key: np.square(np.asarray(weighted_beta_rms_deg(values, np.zeros_like(values), weights), dtype=float))
        for key, values in beta.items()
    }
    selection = {key: int(np.argmin(unary[key])) for key in grouped}
    locked = set(map(str, locked_target_ids))
    order = sorted(grouped, key=lambda value: (hashlib.sha256(f"retry17-icm:{value}".encode()).hexdigest(), value))
    for _ in range(int(maximum_sweeps)):
        changed = False
        for target in order:
            if target in locked:
                continue
            costs = unary[target].copy()
            for neighbour, coefficient in adjacency[target]:
                other = beta[neighbour][selection[neighbour]]
                gap = np.asarray(weighted_beta_rms_deg(beta[target], np.broadcast_to(other, beta[target].shape), weights), dtype=float)
                costs += float(pairwise_lambda) * coefficient * np.minimum(np.square(gap), tau_deg * tau_deg)
            chosen = int(np.argmin(costs))
            changed = changed or chosen != selection[target]
            selection[target] = chosen
        if not changed:
            break
    result = pd.concat([grouped[key].iloc[[selection[key]]] for key in sorted(grouped)], ignore_index=True, sort=False)
    result["teacher"] = "T2_density_normalized"
    result["pairwise_lambda"] = float(pairwise_lambda)
    return result


def teacher_edge_metrics(labels: pd.DataFrame, edges: pd.DataFrame, *, weights: Sequence[float]) -> Mapping[str, float]:
    lookup = labels.set_index("target_id")
    weighted, raw = [], []
    for edge in edges.to_dict("records"):
        left, right = str(edge["left_target_id"]), str(edge["right_target_id"])
        if left not in lookup.index or right not in lookup.index:
            continue
        a = lookup.loc[left, list(BETA_COLUMNS)].to_numpy(float)
        b = lookup.loc[right, list(BETA_COLUMNS)].to_numpy(float)
        weighted.append(float(weighted_beta_rms_deg(a, b, weights)))
        raw.append(float(np.max(np.abs(np.rad2deg(a - b)))))
    return {
        "edge_count": len(weighted),
        "weighted_p95_deg": float(np.percentile(weighted, 95)) if weighted else math.inf,
        "raw_gt7_rate": float(np.mean(np.asarray(raw) > 7.0)) if raw else 1.0,
        "raw_maximum_deg": float(np.max(raw)) if raw else math.inf,
    }


__all__ = [
    "ContinuityPolicy", "annular_coordinates", "continuous_annular_sobol",
    "coverage_distance", "density_normalized_select_t2", "geometric_graph",
    "graph_lcc_fraction", "graph_stretch", "interpolate_profile",
    "local_directional_coverage", "points_inside_profile", "progressive_farthest_fill",
    "teacher_edge_metrics",
]
