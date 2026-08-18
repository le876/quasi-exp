"""Physical-identity helpers and targeted holonomy diagnostics.

Audit schedule identifiers are storage identities.  Repeat perturbations and
cross-phase comparisons instead use canonical physical paths so chart names,
fragment names, and execution phases cannot silently change the experiment.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .canonical import beta_rms_deg
from .canonical_atlas import (
    AtlasCandidate,
    AtlasTaskNode,
    ContinuationAdapter,
)


def _cycle_body(path: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in path)
    if len(values) > 1 and values[0] == values[-1]:
        return values[:-1]
    return values


def _oriented_path(path: Sequence[int], *, cycle: bool) -> tuple[int, ...]:
    values = _cycle_body(path) if cycle else tuple(int(value) for value in path)
    if not cycle or not values:
        return values
    return min(
        values[index:] + values[:index] for index in range(len(values))
    )


def canonical_physical_path(
    path: Sequence[int], *, cycle: bool
) -> tuple[int, ...]:
    """Return a rotation- and direction-invariant physical node path."""

    values = _cycle_body(path) if cycle else tuple(int(value) for value in path)
    if not values:
        return ()
    if cycle:
        forward = _oriented_path(values, cycle=True)
        reverse = _oriented_path(tuple(reversed(values)), cycle=True)
        return min(forward, reverse)
    return min(values, tuple(reversed(values)))


def physical_audit_entity_id(
    patch_id: str, audit_kind: str, path: Sequence[int]
) -> str:
    cycle = "cycle" in str(audit_kind)
    canonical = canonical_physical_path(path, cycle=cycle)
    return f"{patch_id}:{audit_kind}:{','.join(map(str, canonical))}"


def physical_audit_direction(
    registered_path: Sequence[int],
    runtime_direction: Literal["forward", "reverse"] | str,
    *,
    cycle: bool,
) -> Literal["canonical_forward", "canonical_reverse"]:
    """Map a schedule-local direction onto the physical path orientation."""

    values = tuple(int(value) for value in registered_path)
    if str(runtime_direction) == "reverse":
        values = tuple(reversed(values))
    elif str(runtime_direction) != "forward":
        raise ValueError(f"unsupported audit direction: {runtime_direction}")
    oriented = _oriented_path(values, cycle=cycle)
    canonical = canonical_physical_path(values, cycle=cycle)
    return (
        "canonical_forward"
        if oriented == canonical
        else "canonical_reverse"
    )


def physical_repeat_perturbation(
    physical_entity_id: str,
    physical_direction: str,
    repeat_index: int,
    magnitude_rad: float,
) -> np.ndarray:
    if int(repeat_index) == 0:
        return np.zeros(6, dtype=float)
    digest = hashlib.sha256(
        f"{physical_entity_id}:{physical_direction}:{int(repeat_index)}".encode(
            "utf-8"
        )
    ).digest()
    signs = np.asarray(
        [1.0 if digest[index] & 1 else -1.0 for index in range(6)],
        dtype=float,
    )
    return signs * float(magnitude_rad) * int(repeat_index)


def _trajectory_candidate(
    target: AtlasTaskNode,
    beta_rad: np.ndarray,
    residual_mm: float,
    source: AtlasCandidate,
    *,
    prefix_index: int,
) -> AtlasCandidate:
    return AtlasCandidate(
        node_id=target.node_id,
        candidate_id=f"holonomy_prefix_{prefix_index:04d}",
        beta_rad=beta_rad,
        residual_mm=max(0.0, float(residual_mm)),
        min_margin_deg=source.min_margin_deg,
        normalized_min_margin=source.normalized_min_margin,
        posture_cost=float(np.linalg.norm(beta_rad)),
        condition_number=source.condition_number,
        quality=source.quality,
        solver_success=True,
        actual_bounds=True,
        diagnostics={"holonomy_prefix": True},
    )


def replay_cycle_with_prefix_trace(
    path_node_ids: Sequence[int],
    *,
    patch_id: str,
    canonical_by_node: Mapping[int, AtlasCandidate],
    task_nodes: Mapping[int, AtlasTaskNode],
    continuation: ContinuationAdapter,
    start_node_id: int | None = None,
    direction: Literal["forward", "reverse"] = "forward",
    repeat_id: int = 0,
    perturbation_magnitude_rad: float = 0.0,
    kernel_id: str = "baseline",
    environment: Any | None = None,
) -> pd.DataFrame:
    """Replay one physical cycle and retain every transported prefix endpoint.

    Canonical target beta is used only for post-hoc gap measurement.  It is
    never passed to ``continuation`` as a seed or reference.
    """

    body = _cycle_body(path_node_ids)
    if len(body) < 2:
        raise ValueError("holonomy replay requires a cycle with at least two nodes")
    start = int(body[0] if start_node_id is None else start_node_id)
    if start not in body:
        raise ValueError("start_node_id is not part of the registered cycle")
    index = body.index(start)
    ordered = body[index:] + body[:index]
    if direction == "reverse":
        ordered = (ordered[0], *tuple(reversed(ordered[1:])))
    elif direction != "forward":
        raise ValueError(f"unsupported holonomy direction: {direction}")
    closed = (*ordered, ordered[0])
    entity = physical_audit_entity_id(patch_id, "fundamental_cycle", body)
    physical_direction = physical_audit_direction(
        closed, direction, cycle=True
    )
    magnitude = float(perturbation_magnitude_rad)
    perturbation = (
        np.zeros(6, dtype=float)
        if magnitude == 0.0
        else np.sign(magnitude)
        * physical_repeat_perturbation(
            entity,
            physical_direction,
            1,
            abs(magnitude),
        )
    )
    registered_source = canonical_by_node[start]
    source = _trajectory_candidate(
        task_nodes[start],
        registered_source.beta_rad + perturbation,
        registered_source.residual_mm,
        registered_source,
        prefix_index=0,
    )
    records: list[dict[str, Any]] = []
    for prefix_index, (source_node_id, target_node_id) in enumerate(
        zip(closed[:-1], closed[1:]), start=1
    ):
        target = task_nodes[int(target_node_id)]
        outcome = continuation(source, target)
        canonical = canonical_by_node[int(target_node_id)]
        singular_values: list[float] = []
        normalized_condition = math.inf
        active_bounds = 0
        if environment is not None and hasattr(environment, "fk_and_jacobian"):
            _xyz, jacobian = environment.fk_and_jacobian(outcome.beta_rad)
            singular_values = list(
                map(float, np.linalg.svd(np.asarray(jacobian)[0], compute_uv=False))
            )
            if singular_values and singular_values[-1] > 0.0:
                normalized_condition = singular_values[0] / singular_values[-1]
            if hasattr(environment, "bounds"):
                bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
                active_bounds = int(
                    np.sum(
                        np.minimum(
                            outcome.beta_rad - bounds[:, 0],
                            bounds[:, 1] - outcome.beta_rad,
                        )
                        <= 1.0e-9
                    )
                )
        record: dict[str, Any] = {
            "physical_entity_id": entity,
            "start_node_id": start,
            "direction": direction,
            "physical_direction": physical_direction,
            "repeat_id": int(repeat_id),
            "kernel_id": str(kernel_id),
            "prefix_index": int(prefix_index),
            "source_node_id": int(source_node_id),
            "target_node_id": int(target_node_id),
            "geometry_gap_deg": beta_rms_deg(
                outcome.beta_rad, canonical.beta_rad
            ),
            "fk_residual_mm": float(outcome.residual_mm),
            "success": bool(outcome.success),
            "jacobian_singular_values": singular_values,
            "normalized_condition": float(normalized_condition),
            "active_bounds": active_bounds,
            "solver_tier": str(outcome.status),
            "solver_iterations": int(outcome.corrector_iterations),
            "nullspace_step_norm_deg": float(
                np.linalg.norm(outcome.beta_rad - source.beta_rad)
                * 180.0
                / math.pi
            ),
        }
        for beta_index in range(6):
            record[f"beta_{beta_index}"] = float(outcome.beta_rad[beta_index])
            record[f"canonical_beta_{beta_index}"] = float(
                canonical.beta_rad[beta_index]
            )
        records.append(record)
        source = _trajectory_candidate(
            target,
            outcome.beta_rad,
            outcome.residual_mm,
            source,
            prefix_index=prefix_index,
        )
        if not outcome.success:
            break
    return pd.DataFrame.from_records(records)


def audit_cycle_all_starts(
    path_node_ids: Sequence[int],
    *,
    patch_id: str,
    canonical_by_node: Mapping[int, AtlasCandidate],
    task_nodes: Mapping[int, AtlasTaskNode],
    continuation: ContinuationAdapter,
    perturbation_magnitudes_rad: Sequence[float] = (0.0,),
    kernel_id: str = "baseline",
    environment: Any | None = None,
) -> pd.DataFrame:
    frames = []
    body = _cycle_body(path_node_ids)
    for start in body:
        for direction in ("forward", "reverse"):
            for repeat_id, magnitude in enumerate(perturbation_magnitudes_rad):
                frames.append(
                    replay_cycle_with_prefix_trace(
                        body,
                        patch_id=patch_id,
                        canonical_by_node=canonical_by_node,
                        task_nodes=task_nodes,
                        continuation=continuation,
                        start_node_id=start,
                        direction=direction,
                        repeat_id=repeat_id,
                        perturbation_magnitude_rad=float(magnitude),
                        kernel_id=kernel_id,
                        environment=environment,
                    )
                )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def audit_canonical_reset_edges(
    path_node_ids: Sequence[int],
    *,
    patch_id: str,
    canonical_by_node: Mapping[int, AtlasCandidate],
    task_nodes: Mapping[int, AtlasTaskNode],
    continuation: ContinuationAdapter,
    kernel_id: str = "baseline",
) -> pd.DataFrame:
    """Audit each cycle edge from its stored canonical source independently."""

    body = _cycle_body(path_node_ids)
    closed = (*body, body[0])
    records = []
    entity = physical_audit_entity_id(patch_id, "fundamental_cycle", body)
    for direction in ("forward", "reverse"):
        path = closed if direction == "forward" else tuple(reversed(closed))
        for edge_index, (left, right) in enumerate(zip(path[:-1], path[1:])):
            source = canonical_by_node[int(left)]
            outcome = continuation(source, task_nodes[int(right)])
            target = canonical_by_node[int(right)]
            records.append(
                {
                    "physical_entity_id": entity,
                    "edge_index": int(edge_index),
                    "direction": direction,
                    "source_node_id": int(left),
                    "target_node_id": int(right),
                    "kernel_id": str(kernel_id),
                    "geometry_gap_deg": beta_rms_deg(
                        outcome.beta_rad, target.beta_rad
                    ),
                    "fk_residual_mm": float(outcome.residual_mm),
                    "success": bool(outcome.success),
                    "solver_tier": str(outcome.status),
                    "solver_iterations": int(outcome.corrector_iterations),
                }
            )
    return pd.DataFrame.from_records(records)


def _cycle_rank(nodes: set[int], edges: set[tuple[int, int]]) -> int:
    remaining = set(nodes)
    components = 0
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    while remaining:
        components += 1
        queue = [remaining.pop()]
        while queue:
            current = queue.pop()
            neighbours = adjacency[current] & remaining
            remaining -= neighbours
            queue.extend(neighbours)
    return max(0, len(edges) - len(nodes) + components)


def audit_refinement_edge_ablation(
    task_edges: Sequence[tuple[int, int]],
    refinement_edges: Sequence[tuple[int, int]],
    *,
    holonomy_by_edge: Mapping[tuple[int, int], float] | None = None,
) -> pd.DataFrame:
    """Report the exact topology and observed holonomy change per edge ablation."""

    edges = {tuple(sorted(map(int, edge))) for edge in task_edges}
    nodes = {node for edge in edges for node in edge}
    baseline_rank = _cycle_rank(nodes, edges)
    records = []
    observed = {} if holonomy_by_edge is None else {
        tuple(sorted(map(int, edge))): float(value)
        for edge, value in holonomy_by_edge.items()
    }
    for edge in sorted({tuple(sorted(map(int, item))) for item in refinement_edges}):
        retained = edges - {edge}
        records.append(
            {
                "left_node_id": edge[0],
                "right_node_id": edge[1],
                "edge_registered": edge in edges,
                "baseline_cycle_rank": baseline_rank,
                "ablated_cycle_rank": _cycle_rank(nodes, retained),
                "cycle_rank_change": _cycle_rank(nodes, retained) - baseline_rank,
                "holonomy_change_deg": observed.get(edge),
            }
        )
    return pd.DataFrame.from_records(records)


__all__ = [
    "canonical_physical_path",
    "physical_audit_direction",
    "physical_audit_entity_id",
    "physical_repeat_perturbation",
    "replay_cycle_with_prefix_trace",
    "audit_cycle_all_starts",
    "audit_canonical_reset_edges",
    "audit_refinement_edge_ablation",
]
