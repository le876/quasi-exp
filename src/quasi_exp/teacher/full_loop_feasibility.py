"""Strict full-loop inverse-branch discovery primitives for protocol V11.4.

The module deliberately separates three questions that V11.3 mixed together:

* whether root IK solutions are locally connected;
* whether an individual root lineage survives a complete traversal; and
* whether hard-feasible per-phase nodes contain a closed, smooth cycle.

No function in this module pads a candidate layer with an uncorrected seed.
An undersized or empty hard-feasible layer is evidence and remains undersized.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_BETA_WEIGHTS = np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0])


def weighted_rms_gap_deg(
    left_rad: np.ndarray,
    right_rad: np.ndarray,
    *,
    beta_weights: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Return weighted joint-space RMS distance in degrees."""

    left = np.asarray(left_rad, dtype=float)
    right = np.asarray(right_rad, dtype=float)
    weights = np.asarray(
        DEFAULT_BETA_WEIGHTS if beta_weights is None else beta_weights,
        dtype=float,
    ).reshape(6)
    if np.any(weights <= 0.0):
        raise ValueError("beta_weights must contain six positive values")
    delta = np.broadcast_arrays(left, right)[0] - np.broadcast_arrays(left, right)[1]
    return np.rad2deg(
        np.sqrt(np.sum(np.square(delta) * weights, axis=-1) / np.sum(weights))
    )


def _pairwise_gap_deg(
    beta: np.ndarray, beta_weights: np.ndarray | Sequence[float] | None = None
) -> np.ndarray:
    values = np.asarray(beta, dtype=float).reshape(-1, 6)
    return weighted_rms_gap_deg(
        values[:, None, :],
        values[None, :, :],
        beta_weights=beta_weights,
    )


def _component_labels(adjacency: Sequence[set[int]]) -> np.ndarray:
    labels = np.full(len(adjacency), -1, dtype=np.int64)
    component = 0
    for start in range(len(adjacency)):
        if labels[start] >= 0:
            continue
        labels[start] = component
        pending = [start]
        while pending:
            left = pending.pop()
            for right in adjacency[left]:
                if labels[right] < 0:
                    labels[right] = component
                    pending.append(right)
        component += 1
    return labels


def connectivity_sensitivity(
    beta_rad: np.ndarray,
    *,
    thresholds_deg: Sequence[float],
    beta_weights: np.ndarray | Sequence[float] | None = None,
) -> pd.DataFrame:
    """Count epsilon-graph components without calling them physical branches."""

    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    if len(beta) == 0:
        raise ValueError("beta_rad must contain at least one root solution")
    gap = _pairwise_gap_deg(beta, beta_weights)
    rows: list[dict[str, Any]] = []
    for raw_threshold in thresholds_deg:
        threshold = float(raw_threshold)
        if threshold <= 0.0:
            raise ValueError("connectivity thresholds must be positive")
        adjacency = [
            set(np.flatnonzero(gap[index] <= threshold + 1.0e-12).tolist())
            - {index}
            for index in range(len(beta))
        ]
        labels = _component_labels(adjacency)
        rows.append(
            {
                "epsilon_deg": threshold,
                "connected_component_count": int(labels.max() + 1),
                "largest_component_size": int(
                    max(np.bincount(labels), default=0)
                ),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ValidatedRootGraph:
    edge_table: pd.DataFrame
    component_labels: np.ndarray
    connected_component_count: int


def candidate_search_budgets(
    *,
    candidate_cap: int,
    initial_budget_floor: int,
    initial_budget_multiplier: int,
    targeted_cumulative_budgets: Sequence[int],
) -> tuple[int, ...]:
    """Return cumulative search budgets without reducing the initial budget.

    Targeted budgets below ``max(multiplier * K, floor)`` are already covered
    by the registered initial search and must not be run first.
    """

    cap = int(candidate_cap)
    floor = int(initial_budget_floor)
    multiplier = int(initial_budget_multiplier)
    if cap < 1 or floor < 1 or multiplier < 1:
        raise ValueError("candidate cap, floor, and multiplier must be positive")
    initial = max(multiplier * cap, floor)
    targeted = sorted(
        {
            int(value)
            for value in targeted_cumulative_budgets
            if int(value) > initial
        }
    )
    if any(value < 1 for value in targeted_cumulative_budgets):
        raise ValueError("targeted cumulative budgets must be positive")
    return (initial, *targeted)


def validated_mutual_knn_graph(
    beta_rad: np.ndarray,
    *,
    k: int,
    max_edge_deg: float,
    interpolation_fractions: Sequence[float],
    interpolation_is_feasible: Callable[[np.ndarray, np.ndarray, float], bool],
    beta_weights: np.ndarray | Sequence[float] | None = None,
) -> ValidatedRootGraph:
    """Build a mutual-kNN graph and validate each local IK interpolation.

    Validation is deliberately bidirectional because corrector basins can be
    asymmetric even when the geometric segment is the same.
    """

    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    if len(beta) == 0:
        raise ValueError("beta_rad must contain at least one root solution")
    neighbor_count = min(max(int(k), 1), max(len(beta) - 1, 1))
    gap = _pairwise_gap_deg(beta, beta_weights)
    np.fill_diagonal(gap, np.inf)
    nearest = [
        set(np.argsort(gap[index], kind="stable")[:neighbor_count].tolist())
        for index in range(len(beta))
    ]
    adjacency = [set() for _ in range(len(beta))]
    rows: list[dict[str, Any]] = []
    fractions = tuple(float(value) for value in interpolation_fractions)
    if not fractions or any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("interpolation fractions must lie strictly between zero and one")
    for left in range(len(beta)):
        for right in sorted(nearest[left]):
            if right <= left or left not in nearest[right]:
                continue
            distance = float(gap[left, right])
            if distance > float(max_edge_deg) + 1.0e-12:
                continue
            checks = [
                bool(interpolation_is_feasible(beta[left], beta[right], fraction))
                and bool(
                    interpolation_is_feasible(
                        beta[right], beta[left], fraction
                    )
                )
                for fraction in fractions
            ]
            validated = bool(all(checks))
            if validated:
                adjacency[left].add(right)
                adjacency[right].add(left)
                rows.append(
                    {
                        "left_idx": left,
                        "right_idx": right,
                        "distance_deg": distance,
                        "validated": True,
                        "interpolation_fraction_count": len(fractions),
                    }
                )
    labels = _component_labels(adjacency)
    edge_table = pd.DataFrame(
        rows,
        columns=[
            "left_idx",
            "right_idx",
            "distance_deg",
            "validated",
            "interpolation_fraction_count",
        ],
    )
    return ValidatedRootGraph(
        edge_table=edge_table,
        component_labels=labels,
        connected_component_count=int(labels.max() + 1),
    )


def geodesic_farthest_representatives(
    beta_rad: np.ndarray,
    *,
    max_count: int,
    required_indices: Sequence[int] = (),
    beta_weights: np.ndarray | Sequence[float] | None = None,
    geodesic_distance_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Select deterministic farthest-point representatives in joint space."""

    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    requested = min(max(int(max_count), 0), len(beta))
    if requested == 0:
        return np.zeros(0, dtype=np.int64)
    chosen: list[int] = []
    for raw_index in required_indices:
        index = int(raw_index)
        if not 0 <= index < len(beta):
            raise IndexError("required representative index is out of range")
        if index not in chosen:
            chosen.append(index)
    chosen = chosen[:requested]
    if not chosen:
        chosen.append(0)
    gap = (
        _pairwise_gap_deg(beta, beta_weights)
        if geodesic_distance_deg is None
        else np.asarray(geodesic_distance_deg, dtype=float).reshape(
            len(beta), len(beta)
        )
    )
    nearest = np.min(gap[:, chosen], axis=1)
    while len(chosen) < requested:
        nearest[chosen] = -math.inf
        disconnected = np.flatnonzero(np.isposinf(nearest))
        next_index = (
            int(disconnected[0])
            if len(disconnected)
            else int(np.argmax(nearest))
        )
        chosen.append(next_index)
        nearest = np.minimum(nearest, gap[:, next_index])
    return np.asarray(chosen, dtype=np.int64)


@dataclass(frozen=True)
class HardCandidateLayer:
    beta_rad: np.ndarray
    table: pd.DataFrame
    requested_cap: int
    shortfall: int


@dataclass(frozen=True)
class FullLoopCurveClusters:
    """Pairwise full-trajectory distances and connected-component labels."""

    distance_deg: np.ndarray
    component_labels: np.ndarray
    connected_component_count: int


def cluster_full_loop_cycles(
    cycles_rad: Sequence[np.ndarray],
    *,
    threshold_deg: float,
    beta_weights: np.ndarray | Sequence[float] | None = None,
) -> FullLoopCurveClusters:
    """Cluster complete cycles by their phase-wise weighted-gap P95.

    All cycles must use the same phase inventory.  The connected components of
    the threshold graph are reported; a root-point distance is never used.
    """

    cycles = [np.asarray(value, dtype=float).reshape(-1, 6) for value in cycles_rad]
    if not cycles:
        return FullLoopCurveClusters(
            distance_deg=np.zeros((0, 0), dtype=float),
            component_labels=np.zeros(0, dtype=np.int64),
            connected_component_count=0,
        )
    phase_count = len(cycles[0])
    if phase_count < 1 or any(len(value) != phase_count for value in cycles):
        raise ValueError("all cycles must contain the same non-empty phase inventory")
    threshold = float(threshold_deg)
    if threshold <= 0.0:
        raise ValueError("curve cluster threshold must be positive")
    count = len(cycles)
    distance = np.zeros((count, count), dtype=float)
    adjacency = [set() for _ in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            phase_gap = weighted_rms_gap_deg(
                cycles[left],
                cycles[right],
                beta_weights=beta_weights,
            )
            value = float(np.percentile(phase_gap, 95))
            distance[left, right] = value
            distance[right, left] = value
            if value <= threshold + 1.0e-12:
                adjacency[left].add(right)
                adjacency[right].add(left)
    labels = _component_labels(adjacency)
    return FullLoopCurveClusters(
        distance_deg=distance,
        component_labels=labels,
        connected_component_count=int(labels.max() + 1),
    )


def filter_hard_feasible_candidates(
    beta_rad: np.ndarray,
    records: pd.DataFrame,
    *,
    cap: int,
    residual_max_mm: float,
    margin_min_deg: float,
    distinct_threshold_deg: float,
    beta_weights: np.ndarray | Sequence[float] | None = None,
) -> HardCandidateLayer:
    """Keep only corrected, bounded, distinct, hard-feasible candidates."""

    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    table = records.reset_index(drop=True).copy()
    if len(beta) != len(table):
        raise ValueError("beta_rad and records must have identical row counts")
    required = {
        "corrector_success",
        "residual_mm",
        "joint_margin_deg",
        "within_bounds",
    }
    missing = required - set(table)
    if missing:
        raise ValueError(f"candidate records missing columns: {sorted(missing)}")
    requested = int(cap)
    if requested < 1:
        raise ValueError("cap must be positive")
    hard = (
        table["corrector_success"].astype(bool).to_numpy()
        & table["within_bounds"].astype(bool).to_numpy()
        & (table["residual_mm"].to_numpy(float) <= float(residual_max_mm))
        & (table["joint_margin_deg"].to_numpy(float) >= float(margin_min_deg))
        & np.isfinite(beta).all(axis=1)
    )
    kept: list[int] = []
    for index in np.flatnonzero(hard):
        if any(
            float(
                weighted_rms_gap_deg(
                    beta[index],
                    beta[previous],
                    beta_weights=beta_weights,
                )
            )
            < float(distinct_threshold_deg) - 1.0e-12
            for previous in kept
        ):
            continue
        kept.append(int(index))
        if len(kept) >= requested:
            break
    selected = table.iloc[kept].copy().reset_index(drop=True)
    selected["source_candidate_idx"] = np.asarray(kept, dtype=np.int64)
    return HardCandidateLayer(
        beta_rad=beta[kept].copy(),
        table=selected,
        requested_cap=requested,
        shortfall=requested - len(kept),
    )


def rank_viability(table: pd.DataFrame) -> pd.DataFrame:
    """Rank root lineages by the preregistered viability-first lexicographic key."""

    order = [
        "completed_360",
        "strict_feasible_ratio",
        "joint_margin_min_deg",
        "residual_max_mm",
        "residual_p95_mm",
        "velocity_p95_deg",
        "acceleration_p95_deg",
        "posture_cost",
        "kappa_p95",
    ]
    missing = set(order) - set(table)
    if missing:
        raise ValueError(f"viability table missing columns: {sorted(missing)}")
    ranked = table.copy()
    ranked["_original_order"] = np.arange(len(ranked), dtype=np.int64)
    ranked = ranked.sort_values(
        [*order, "_original_order"],
        ascending=[False, False, False, True, True, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked["viability_rank"] = np.arange(1, len(ranked) + 1, dtype=np.int64)
    return ranked.drop(columns="_original_order")


def _layer_edges(
    left: np.ndarray,
    right: np.ndarray,
    *,
    beta_weights: np.ndarray,
    edge_limit_deg: float,
) -> list[np.ndarray]:
    gap = weighted_rms_gap_deg(
        np.asarray(left)[:, None, :],
        np.asarray(right)[None, :, :],
        beta_weights=beta_weights,
    )
    return [
        np.flatnonzero(gap[index] <= float(edge_limit_deg) + 1.0e-12)
        for index in range(len(left))
    ]


@dataclass(frozen=True)
class CycleSolution:
    success: bool
    beta_rad: np.ndarray
    layer_candidate_indices: np.ndarray
    cost: float
    root_candidate_idx: int | None
    edge_limit_deg: float
    reason: str
    top_cycles: tuple[Mapping[str, Any], ...]
    state_pruning_count: int = 0
    max_states_per_root: int | None = None


def solve_sparse_cycle(
    candidate_layers: Sequence[np.ndarray],
    residual_layers_mm: Sequence[np.ndarray],
    *,
    beta_weights: np.ndarray | Sequence[float],
    edge_limit_deg: float,
    lambda_velocity: float,
    lambda_acceleration: float,
    top_m: int,
    max_states_per_root: int | None = None,
) -> CycleSolution:
    """Solve a hard-node cyclic graph using sparse second-order edge states.

    Every initial root edge is enumerated.  At each phase, states with the same
    ``(previous, current)`` pair retain only their lowest-cost history.  Closure
    checks include both final→root velocity and the two accelerations that cross
    the cyclic seam.
    """

    layers = [np.asarray(value, dtype=float).reshape(-1, 6) for value in candidate_layers]
    residuals = [np.asarray(value, dtype=float).reshape(-1) for value in residual_layers_mm]
    if len(layers) < 3 or len(layers) != len(residuals):
        raise ValueError("candidate and residual layers must align and contain N >= 3")
    if any(len(beta) == 0 for beta in layers):
        return CycleSolution(
            False,
            np.zeros((0, 6)),
            np.zeros(0, dtype=np.int64),
            math.inf,
            None,
            float(edge_limit_deg),
            "empty_feasible_layer",
            (),
            0,
            max_states_per_root,
        )
    if any(len(beta) != len(error) for beta, error in zip(layers, residuals)):
        raise ValueError("every residual layer must align with its candidate layer")
    weights = np.asarray(beta_weights, dtype=float).reshape(6)
    edges = [
        _layer_edges(
            layers[index],
            layers[(index + 1) % len(layers)],
            beta_weights=weights,
            edge_limit_deg=float(edge_limit_deg),
        )
        for index in range(len(layers))
    ]
    transition_costs: list[np.ndarray] = []
    for phase in range(len(layers)):
        gap = weighted_rms_gap_deg(
            layers[phase][:, None, :],
            layers[(phase + 1) % len(layers)][None, :, :],
            beta_weights=weights,
        )
        transition_costs.append(
            np.where(
                gap <= float(edge_limit_deg) + 1.0e-12,
                float(lambda_velocity) * np.square(gap),
                np.inf,
            )
        )
    # Exact first-order reachability prevents beam pruning in the second-order
    # optimizer from turning an existing cycle into a false "no cycle" result.
    completed: list[dict[str, Any]] = []
    for start in range(len(layers[0])):
        cost = np.full(len(layers[0]), np.inf)
        cost[start] = float(residuals[0][start] ** 2)
        parents: list[np.ndarray] = []
        for phase in range(1, len(layers)):
            values = cost[:, None] + transition_costs[phase - 1]
            parent = np.argmin(values, axis=0).astype(np.int64)
            cost = (
                values[parent, np.arange(values.shape[1])]
                + np.square(residuals[phase])
            )
            parents.append(parent)
        total = cost + transition_costs[-1][:, start]
        if not np.isfinite(total).any():
            continue
        end = int(np.argmin(total))
        indices = [end]
        current = end
        for parent in reversed(parents):
            current = int(parent[current])
            indices.append(current)
        indices = list(reversed(indices))
        beta_path = np.vstack(
            [layers[phase][indices[phase]] for phase in range(len(layers))]
        )
        acceleration = weighted_rms_gap_deg(
            np.roll(beta_path, -1, axis=0)
            - 2.0 * beta_path
            + np.roll(beta_path, 1, axis=0),
            np.zeros_like(beta_path),
            beta_weights=weights,
        )
        completed.append(
            {
                "cost": float(total[end])
                + float(lambda_acceleration)
                * float(np.sum(np.square(acceleration))),
                "root_candidate_idx": start,
                "layer_candidate_indices": tuple(indices),
                "selection_mode": "exact_first_order_reachability",
            }
        )
    pruning_count = 0
    for start in range(len(layers[0])):
        states: dict[tuple[int, int, int], tuple[float, tuple[int, ...]]] = {}
        for raw_second in edges[0][start]:
            second = int(raw_second)
            velocity = float(
                weighted_rms_gap_deg(
                    layers[0][start], layers[1][second], beta_weights=weights
                )
            )
            states[(second, start, second)] = (
                    float(residuals[0][start] ** 2 + residuals[1][second] ** 2)
                    + float(lambda_velocity) * velocity**2,
                    (start, second),
                )
        for phase in range(1, len(layers) - 1):
            next_states: dict[
                tuple[int, int, int], tuple[float, tuple[int, ...]]
            ] = {}
            for (first_second, previous, current), (cost, path) in states.items():
                for raw_next in edges[phase][current]:
                    next_index = int(raw_next)
                    velocity = float(
                        weighted_rms_gap_deg(
                            layers[phase][current],
                            layers[phase + 1][next_index],
                            beta_weights=weights,
                        )
                    )
                    acceleration = float(
                        weighted_rms_gap_deg(
                            layers[phase + 1][next_index]
                            - 2.0 * layers[phase][current]
                            + layers[phase - 1][previous],
                            np.zeros(6),
                            beta_weights=weights,
                        )
                    )
                    candidate_cost = (
                        cost
                        + float(residuals[phase + 1][next_index] ** 2)
                        + float(lambda_velocity) * velocity**2
                        + float(lambda_acceleration) * acceleration**2
                    )
                    key = (first_second, current, next_index)
                    if key not in next_states or candidate_cost < next_states[key][0]:
                        next_states[key] = (candidate_cost, (*path, next_index))
            if (
                max_states_per_root is not None
                and len(next_states) > int(max_states_per_root)
            ):
                retained = sorted(
                    next_states.items(),
                    key=lambda item: (
                        float(item[1][0]),
                        item[0],
                        item[1][1],
                    ),
                )[: int(max_states_per_root)]
                pruning_count += len(next_states) - len(retained)
                next_states = dict(retained)
            states = next_states
            if not states:
                break
        for (_first_second, penultimate, last), (cost, path) in states.items():
            if start not in edges[-1][last]:
                continue
            seam_velocity = float(
                weighted_rms_gap_deg(
                    layers[-1][last], layers[0][start], beta_weights=weights
                )
            )
            seam_accel_last = float(
                weighted_rms_gap_deg(
                    layers[0][start]
                    - 2.0 * layers[-1][last]
                    + layers[-2][penultimate],
                    np.zeros(6),
                    beta_weights=weights,
                )
            )
            seam_accel_root = float(
                weighted_rms_gap_deg(
                    layers[1][path[1]]
                    - 2.0 * layers[0][start]
                    + layers[-1][last],
                    np.zeros(6),
                    beta_weights=weights,
                )
            )
            total = (
                cost
                + float(lambda_velocity) * seam_velocity**2
                + float(lambda_acceleration)
                * (seam_accel_last**2 + seam_accel_root**2)
            )
            completed.append(
                {
                    "cost": float(total),
                    "root_candidate_idx": start,
                    "layer_candidate_indices": tuple(int(value) for value in path),
                    "selection_mode": "sparse_second_order_edge_state",
                }
            )
    if not completed:
        return CycleSolution(
            False,
            np.zeros((0, 6)),
            np.zeros(0, dtype=np.int64),
            math.inf,
            None,
            float(edge_limit_deg),
            "no_closed_cycle",
            (),
            pruning_count,
            max_states_per_root,
        )
    completed.sort(
        key=lambda row: (
            float(row["cost"]),
            int(row["root_candidate_idx"]),
            tuple(row["layer_candidate_indices"]),
        )
    )
    best = completed[0]
    indices = np.asarray(best["layer_candidate_indices"], dtype=np.int64)
    beta = np.vstack(
        [layers[phase][indices[phase]] for phase in range(len(layers))]
    )
    public = tuple(
        {
            "cost": float(row["cost"]),
            "root_candidate_idx": int(row["root_candidate_idx"]),
            "layer_candidate_indices": list(row["layer_candidate_indices"]),
            "selection_mode": str(row["selection_mode"]),
        }
        for row in completed[: max(int(top_m), 1)]
    )
    return CycleSolution(
        True,
        beta,
        indices,
        float(best["cost"]),
        int(best["root_candidate_idx"]),
        float(edge_limit_deg),
        "success",
        public,
        pruning_count,
        max_states_per_root,
    )


def strict_cycle_metrics(
    beta_rad: np.ndarray,
    *,
    target_xyz_m: np.ndarray,
    achieved_xyz_m: np.ndarray,
    joint_margin_deg: np.ndarray,
    beta_weights: np.ndarray | Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute the preregistered hard cycle metrics with cyclic differences."""

    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    target = np.asarray(target_xyz_m, dtype=float).reshape(len(beta), 3)
    achieved = np.asarray(achieved_xyz_m, dtype=float).reshape(len(beta), 3)
    margin = np.asarray(joint_margin_deg, dtype=float).reshape(len(beta))
    residual = np.linalg.norm(achieved - target, axis=1) * 1000.0
    velocity = weighted_rms_gap_deg(
        np.roll(beta, -1, axis=0), beta, beta_weights=beta_weights
    )
    acceleration = weighted_rms_gap_deg(
        np.roll(beta, -1, axis=0) - 2.0 * beta + np.roll(beta, 1, axis=0),
        np.zeros_like(beta),
        beta_weights=beta_weights,
    )
    return {
        "residual_p95_mm": float(np.percentile(residual, 95)),
        "residual_max_mm": float(np.max(residual)),
        "joint_margin_min_deg": float(np.min(margin)),
        "delta_beta_rms_p95_deg": float(np.percentile(velocity, 95)),
        "delta_beta_rms_max_deg": float(np.max(velocity)),
        "acceleration_beta_rms_p95_deg": float(
            np.percentile(acceleration, 95)
        ),
        "seam_beta_rms_deg": float(velocity[-1]),
    }


@dataclass(frozen=True)
class StrictCycleGate:
    residual_p95_mm: float = 1.0
    residual_max_mm: float = 3.0
    joint_margin_min_deg: float = 1.5
    delta_beta_rms_p95_deg: float = 1.0
    delta_beta_rms_max_deg: float = 2.0
    acceleration_beta_rms_p95_deg: float = 0.5
    seam_beta_rms_deg: float = 1.0

    def evaluate(self, metrics: Mapping[str, float]) -> dict[str, Any]:
        required = set(self.__dataclass_fields__)
        missing = required - set(metrics)
        if missing:
            raise ValueError(f"strict cycle metrics missing: {sorted(missing)}")
        checks = {
            "residual_p95": float(metrics["residual_p95_mm"])
            <= self.residual_p95_mm,
            "residual_max": float(metrics["residual_max_mm"])
            <= self.residual_max_mm,
            "joint_margin": float(metrics["joint_margin_min_deg"])
            >= self.joint_margin_min_deg,
            "velocity_p95": float(metrics["delta_beta_rms_p95_deg"])
            <= self.delta_beta_rms_p95_deg,
            "velocity_max": float(metrics["delta_beta_rms_max_deg"])
            <= self.delta_beta_rms_max_deg,
            "acceleration_p95": float(
                metrics["acceleration_beta_rms_p95_deg"]
            )
            <= self.acceleration_beta_rms_p95_deg,
            "seam": float(metrics["seam_beta_rms_deg"]) <= self.seam_beta_rms_deg,
        }
        normalized = {name: bool(value) for name, value in checks.items()}
        return {"checks": normalized, "gate_pass": bool(all(normalized.values()))}


def classify_full_loop_outcome(
    *,
    strict_cycle_pass: bool,
    empty_feasible_layer_indices: Sequence[int],
    targeted_search_complete: bool,
    diagnostic_relaxed_cycle_pass: bool,
) -> str:
    """Return preregistered decision boundary A/B/C/D."""

    if bool(strict_cycle_pass):
        return "A"
    if list(empty_feasible_layer_indices):
        return "C" if bool(targeted_search_complete) else "C_pending_targeted_search"
    if bool(diagnostic_relaxed_cycle_pass):
        return "D"
    return "B"


def formal_decision(*, numerical_outcome: str, audit_pass: bool) -> str:
    """Combine numerical discovery and audit without corrupting A/B/C/D.

    An audit failure after a strict numerical cycle is its own fail-closed
    decision.  It is not evidence that the cycle required relaxed gates.
    """

    outcome = str(numerical_outcome)
    allowed = {"A", "B", "C", "C_pending_targeted_search", "D"}
    if outcome not in allowed:
        raise ValueError(f"unknown numerical outcome: {outcome}")
    if outcome == "A":
        return "A" if bool(audit_pass) else "A_AUDIT_FAILED"
    return outcome
