"""Topology-preserving diagnostics for finite cyclic IK candidate layers.

The helpers in this module deliberately know nothing about the IK solver.  They
only answer two questions:

1. does the finite layered graph contain a cycle that visits every phase under
   the frozen transition threshold; and
2. if not, from which reachable frontier candidates should a caller continue
   the numerical branch in order to repair the discretisation gap?

Keeping graph reachability separate from numerical enrichment makes the final
certificate independent of the dynamic-programming objective used to rank
otherwise feasible cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


def _pairwise_rms_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs = np.asarray(left, dtype=float).reshape(-1, 6)
    rhs = np.asarray(right, dtype=float).reshape(-1, 6)
    delta = lhs[:, None, :] - rhs[None, :, :]
    return np.rad2deg(np.sqrt(np.mean(np.square(delta), axis=2)))


@dataclass(frozen=True)
class BridgeFrontier:
    """One deterministic predictor-corrector seed at a failed graph frontier."""

    source_phase_idx: int
    destination_phase_idx: int
    source_candidate_idx: int
    direction: str
    nearest_existing_gap_deg: float
    root_candidate_idx: int


@dataclass(frozen=True)
class CyclicReachability:
    """Root-identity-preserving reachability evidence for a layered cycle."""

    root_phase_idx: int
    closed_cycle_exists: bool
    compatible_edge_counts: tuple[int, ...]
    minimum_edge_gaps_deg: tuple[float, ...]
    forward_reachable_counts: tuple[int, ...]
    backward_reachable_counts: tuple[int, ...]
    bidirectional_intersection_counts: tuple[int, ...]
    closed_root_candidate_indices: tuple[int, ...]
    frontiers: tuple[BridgeFrontier, ...]

    def to_report(self) -> dict[str, Any]:
        zero_edges = [
            int(index)
            for index, count in enumerate(self.compatible_edge_counts)
            if int(count) == 0
        ]
        return {
            "root_phase_idx": int(self.root_phase_idx),
            "closed_cycle_exists": bool(self.closed_cycle_exists),
            "compatible_edge_counts": list(self.compatible_edge_counts),
            "minimum_edge_gaps_deg": list(self.minimum_edge_gaps_deg),
            "zero_compatible_edge_phase_indices": zero_edges,
            "forward_reachable_counts": list(
                self.forward_reachable_counts
            ),
            "backward_reachable_counts": list(
                self.backward_reachable_counts
            ),
            "bidirectional_intersection_counts": list(
                self.bidirectional_intersection_counts
            ),
            "closed_root_candidate_indices": list(
                self.closed_root_candidate_indices
            ),
            "frontiers": [
                {
                    "source_phase_idx": int(item.source_phase_idx),
                    "destination_phase_idx": int(
                        item.destination_phase_idx
                    ),
                    "source_candidate_idx": int(
                        item.source_candidate_idx
                    ),
                    "direction": str(item.direction),
                    "nearest_existing_gap_deg": float(
                        item.nearest_existing_gap_deg
                    ),
                    "root_candidate_idx": int(item.root_candidate_idx),
                }
                for item in self.frontiers
            ],
        }


def analyze_hard_cyclic_reachability(
    candidate_layers: Sequence[np.ndarray],
    *,
    max_transition_deg: float,
    frontier_sources_per_root: int = 2,
) -> CyclicReachability:
    """Analyze a complete cyclic layered graph without pruning root identity.

    Forward and backward relations retain the root-candidate row.  Their
    intersection therefore means that the *same* root candidate can reach a
    phase and return from it, avoiding the common false positive caused by
    unioning incompatible branch fragments.
    """

    if not candidate_layers:
        raise ValueError("candidate_layers must be non-empty")
    layers = [
        np.asarray(values, dtype=float).reshape(-1, 6)
        for values in candidate_layers
    ]
    if any(len(layer) == 0 for layer in layers):
        raise ValueError("all candidate layers must be non-empty")
    if any(not np.isfinite(layer).all() for layer in layers):
        raise ValueError("candidate layers must be finite")
    limit = float(max_transition_deg)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_transition_deg must be finite and positive")
    source_limit = int(frontier_sources_per_root)
    if source_limit < 1:
        raise ValueError("frontier_sources_per_root must be positive")

    phase_count = len(layers)
    gaps = [
        _pairwise_rms_deg(layers[index], layers[(index + 1) % phase_count])
        for index in range(phase_count)
    ]
    edges = [gap <= limit for gap in gaps]
    edge_counts = tuple(int(np.sum(edge)) for edge in edges)
    minimum_gaps = tuple(float(np.min(gap)) for gap in gaps)
    root_phase = min(
        range(phase_count),
        key=lambda index: (len(layers[index]), int(index)),
    )
    root_count = len(layers[root_phase])

    forward: list[np.ndarray | None] = [None] * phase_count
    backward: list[np.ndarray | None] = [None] * phase_count
    forward[root_phase] = np.eye(root_count, dtype=bool)
    backward[root_phase] = np.eye(root_count, dtype=bool)

    for offset in range(1, phase_count):
        previous = (root_phase + offset - 1) % phase_count
        current = (root_phase + offset) % phase_count
        forward[current] = (
            forward[previous].astype(np.int8)
            @ edges[previous].astype(np.int8)
        ) > 0

    for offset in range(1, phase_count):
        current = (root_phase - offset + 1) % phase_count
        previous = (root_phase - offset) % phase_count
        backward[previous] = (
            backward[current].astype(np.int8)
            @ edges[previous].T.astype(np.int8)
        ) > 0

    forward_arrays = [
        np.asarray(value, dtype=bool) for value in forward
    ]
    backward_arrays = [
        np.asarray(value, dtype=bool) for value in backward
    ]
    intersections = [
        lhs & rhs
        for lhs, rhs in zip(forward_arrays, backward_arrays)
    ]
    probe_phase = (root_phase + 1) % phase_count
    closed_roots = tuple(
        int(index)
        for index in np.flatnonzero(
            np.any(intersections[probe_phase], axis=1)
        )
    )

    frontier_candidates: list[BridgeFrontier] = []
    if not closed_roots:
        # A forward source and backward-returnable destination with the same
        # root row would close the corresponding frontier if an admissible
        # continuation candidate were added.  Keep only a small stable set of
        # sources; later rounds propagate the repaired branch farther.
        for phase_idx in range(phase_count):
            destination = (phase_idx + 1) % phase_count
            edge_gap = gaps[phase_idx]
            for root_candidate in range(root_count):
                left = np.flatnonzero(
                    forward_arrays[phase_idx][root_candidate]
                )
                right = np.flatnonzero(
                    backward_arrays[destination][root_candidate]
                )
                if not len(left) or not len(right):
                    continue
                subgap = edge_gap[np.ix_(left, right)]
                order = np.argsort(
                    np.min(subgap, axis=1), kind="stable"
                )[:source_limit]
                for relative in order:
                    frontier_candidates.append(
                        BridgeFrontier(
                            source_phase_idx=int(phase_idx),
                            destination_phase_idx=int(destination),
                            source_candidate_idx=int(left[int(relative)]),
                            direction="forward",
                            nearest_existing_gap_deg=float(
                                np.min(subgap[int(relative)])
                            ),
                            root_candidate_idx=int(root_candidate),
                        )
                    )

                # Symmetric continuation is essential near folds: correct a
                # returnable right candidate back onto the left phase rather
                # than assuming a single predictor direction.
                reverse_order = np.argsort(
                    np.min(subgap, axis=0), kind="stable"
                )[:source_limit]
                for relative in reverse_order:
                    frontier_candidates.append(
                        BridgeFrontier(
                            source_phase_idx=int(destination),
                            destination_phase_idx=int(phase_idx),
                            source_candidate_idx=int(right[int(relative)]),
                            direction="backward",
                            nearest_existing_gap_deg=float(
                                np.min(subgap[:, int(relative)])
                            ),
                            root_candidate_idx=int(root_candidate),
                        )
                    )

        # When one side of a root-consistent frontier is not reachable yet,
        # propagate the first non-empty reachable set across its collapse
        # edge.  The symmetric backward rule grows the return path toward it.
        if not frontier_candidates:
            for phase_idx in range(phase_count):
                destination = (phase_idx + 1) % phase_count
                edge_gap = gaps[phase_idx]
                for root_candidate in range(root_count):
                    forward_left = np.flatnonzero(
                        forward_arrays[phase_idx][root_candidate]
                    )
                    forward_right = np.flatnonzero(
                        forward_arrays[destination][root_candidate]
                    )
                    if len(forward_left) and not len(forward_right):
                        order = sorted(
                            map(int, forward_left),
                            key=lambda index: (
                                float(np.min(edge_gap[index])),
                                index,
                            ),
                        )[:source_limit]
                        frontier_candidates.extend(
                            BridgeFrontier(
                                source_phase_idx=int(phase_idx),
                                destination_phase_idx=int(destination),
                                source_candidate_idx=int(index),
                                direction="forward",
                                nearest_existing_gap_deg=float(
                                    np.min(edge_gap[index])
                                ),
                                root_candidate_idx=int(root_candidate),
                            )
                            for index in order
                        )
                    backward_right = np.flatnonzero(
                        backward_arrays[destination][root_candidate]
                    )
                    backward_left = np.flatnonzero(
                        backward_arrays[phase_idx][root_candidate]
                    )
                    if len(backward_right) and not len(backward_left):
                        order = sorted(
                            map(int, backward_right),
                            key=lambda index: (
                                float(np.min(edge_gap[:, index])),
                                index,
                            ),
                        )[:source_limit]
                        frontier_candidates.extend(
                            BridgeFrontier(
                                source_phase_idx=int(destination),
                                destination_phase_idx=int(phase_idx),
                                source_candidate_idx=int(index),
                                direction="backward",
                                nearest_existing_gap_deg=float(
                                    np.min(edge_gap[:, index])
                                ),
                                root_candidate_idx=int(root_candidate),
                            )
                            for index in order
                        )

        # Last-resort seed for a completely disconnected root relation: use
        # zero-edge layers first, then the largest minimum gap.  Selecting the
        # smallest gap here would spend enrichment on already compatible
        # edges and was the exact failure mode this diagnostic is meant to
        # avoid.
        if not frontier_candidates:
            phase_order = sorted(
                range(phase_count),
                key=lambda index: (
                    0 if edge_counts[index] == 0 else 1,
                    -minimum_gaps[index],
                    int(index),
                ),
            )
            for phase_idx in phase_order[: min(8, phase_count)]:
                destination = (phase_idx + 1) % phase_count
                edge_gap = gaps[phase_idx]
                left_idx, right_idx = np.unravel_index(
                    int(np.argmin(edge_gap)), edge_gap.shape
                )
                frontier_candidates.extend(
                    [
                        BridgeFrontier(
                            source_phase_idx=int(phase_idx),
                            destination_phase_idx=int(destination),
                            source_candidate_idx=int(left_idx),
                            direction="forward",
                            nearest_existing_gap_deg=float(
                                edge_gap[left_idx, right_idx]
                            ),
                            root_candidate_idx=-1,
                        ),
                        BridgeFrontier(
                            source_phase_idx=int(destination),
                            destination_phase_idx=int(phase_idx),
                            source_candidate_idx=int(right_idx),
                            direction="backward",
                            nearest_existing_gap_deg=float(
                                edge_gap[left_idx, right_idx]
                            ),
                            root_candidate_idx=-1,
                        ),
                    ]
                )

    unique: dict[tuple[int, int, int, str], BridgeFrontier] = {}
    for item in sorted(
        frontier_candidates,
        key=lambda value: (
            value.nearest_existing_gap_deg,
            value.source_phase_idx,
            value.destination_phase_idx,
            value.source_candidate_idx,
            value.direction,
            value.root_candidate_idx,
        ),
    ):
        key = (
            item.source_phase_idx,
            item.destination_phase_idx,
            item.source_candidate_idx,
            item.direction,
        )
        unique.setdefault(key, item)

    return CyclicReachability(
        root_phase_idx=int(root_phase),
        closed_cycle_exists=bool(closed_roots),
        compatible_edge_counts=edge_counts,
        minimum_edge_gaps_deg=minimum_gaps,
        forward_reachable_counts=tuple(
            int(np.sum(value)) for value in forward_arrays
        ),
        backward_reachable_counts=tuple(
            int(np.sum(value)) for value in backward_arrays
        ),
        bidirectional_intersection_counts=tuple(
            int(np.sum(value)) for value in intersections
        ),
        closed_root_candidate_indices=closed_roots,
        frontiers=tuple(unique.values()),
    )


def independently_certify_hard_cyclic_path(
    beta_rad: np.ndarray,
    target_xyz_m: np.ndarray,
    achieved_xyz_m: np.ndarray,
    bounds_rad: np.ndarray,
    *,
    residual_p95_mm: float,
    residual_max_mm: float,
    gold_margin_deg: float,
    max_transition_deg: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Recompute every hard node and edge check without using DP metadata."""

    beta = np.asarray(beta_rad, dtype=float)
    target = np.asarray(target_xyz_m, dtype=float)
    achieved = np.asarray(achieved_xyz_m, dtype=float)
    bounds = np.asarray(bounds_rad, dtype=float).reshape(6, 2)
    if (
        beta.ndim != 2
        or beta.shape[1] != 6
        or target.shape != (len(beta), 3)
        or achieved.shape != target.shape
        or not np.isfinite(beta).all()
        or not np.isfinite(target).all()
        or not np.isfinite(achieved).all()
    ):
        raise ValueError("path certificate inputs must be finite and aligned")
    residual = np.linalg.norm(achieved - target, axis=1) * 1000.0
    margin = np.rad2deg(
        np.min(
            np.minimum(
                beta - bounds[:, 0][None, :],
                bounds[:, 1][None, :] - beta,
            ),
            axis=1,
        )
    )
    transition = np.rad2deg(
        np.sqrt(
            np.mean(
                np.square(np.roll(beta, -1, axis=0) - beta), axis=1
            )
        )
    )
    checks = {
        "all_values_finite": True,
        "actual_mechanical_bounds": bool(
            np.all(beta >= bounds[:, 0][None, :])
            and np.all(beta <= bounds[:, 1][None, :])
        ),
        "gold_margin": bool(np.min(margin) >= float(gold_margin_deg)),
        "residual_p95": bool(
            np.percentile(residual, 95) <= float(residual_p95_mm)
        ),
        "residual_max": bool(
            np.max(residual) <= float(residual_max_mm)
        ),
        "all_phase_edges_at_most_limit": bool(
            np.max(transition) <= float(max_transition_deg)
        ),
        "seam_edge_at_most_limit": bool(
            transition[-1] <= float(max_transition_deg)
        ),
    }
    worst_edge = int(np.argmax(transition))
    worst_residual = int(np.argmax(residual))
    minimum_margin = int(np.argmin(margin))
    report = {
        "checks": checks,
        "gate_pass": bool(all(checks.values())),
        "phase_count": int(len(beta)),
        "edge_count": int(len(transition)),
        "residual_p95_mm": float(np.percentile(residual, 95)),
        "residual_max_mm": float(np.max(residual)),
        "residual_argmax_phase_idx": worst_residual,
        "minimum_joint_margin_deg": float(np.min(margin)),
        "margin_argmin_phase_idx": minimum_margin,
        "transition_p95_deg": float(np.percentile(transition, 95)),
        "transition_max_deg": float(np.max(transition)),
        "transition_argmax_source_phase_idx": worst_edge,
        "transition_argmax_destination_phase_idx": int(
            (worst_edge + 1) % len(beta)
        ),
        "seam_transition_deg": float(transition[-1]),
    }
    detail = {
        "certified_residual_mm": residual,
        "certified_joint_margin_deg": margin,
        "certified_transition_to_next_deg": transition,
    }
    return report, detail
