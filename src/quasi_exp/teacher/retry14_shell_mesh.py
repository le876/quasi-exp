"""Pure, deterministic shell-mesh contracts for BACRA V14.3R retry14.

The retry14 runner owns numerical continuation and artifact I/O.  This module
only registers the quotient-space coverage denominator and verifies/combines
the labels and short solver segments supplied by that runner.  Keeping the
mesh model here makes it impossible for a row-count shortcut to masquerade as
coverage progress.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.teacher.retry12_symmetry import (
    BETA_COLUMNS,
    SYMMETRY_BETA_SIGNS,
    XYZ_COLUMNS,
    stable_id,
    transform_beta,
    transform_xyz,
)


SHELL_CELL_COLUMNS = [
    "cell_id", "u_index", "rho_index", "u_center_mm", "rho_center_mm",
    "proposal_count", "angle_bin_count", "coverage_ring_candidate",
    "trajectory_ring_candidate", "required_shell_cell", "one_cell_hole",
]


def _xyz(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, XYZ_COLUMNS].to_numpy(float)


def _cell_id(u_index: int, rho_index: int) -> str:
    return f"ur:{int(u_index)}:{int(rho_index)}"


def _as_cell_tuple(value: object) -> tuple[int, int]:
    left, right = str(value).split(":")[1:]
    return int(left), int(right)


def quotient_cylindrical_proposals(
    proposals: pd.DataFrame,
    *,
    zero_xyz_m: Sequence[float],
    axial_step_mm: float = 15.0,
    radial_step_mm: float = 15.0,
    angle_bins: int = 8,
) -> pd.DataFrame:
    """Return xyz-only proposals in the fundamental-quadrant cylindrical frame.

    Any incoming beta columns are deliberately ignored and are not copied to
    the returned registry.  The output is deterministic even if the source
    frame contains duplicate IDs or arbitrary input ordering.
    """

    if axial_step_mm <= 0 or radial_step_mm <= 0 or angle_bins <= 0:
        raise ValueError("shell steps and angle_bins must be positive")
    required = set(XYZ_COLUMNS)
    missing = required.difference(proposals.columns)
    if missing:
        raise ValueError(f"proposal xyz columns missing: {sorted(missing)}")
    zero = np.asarray(zero_xyz_m, dtype=float).reshape(3)
    xyz = _xyz(proposals).copy()
    xyz[:, 1:] = np.abs(xyz[:, 1:])
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    source_index = proposals.index.to_numpy()[finite]
    u_mm = (zero[0] - xyz[:, 0]) * 1000.0
    rho_mm = np.hypot(xyz[:, 1], xyz[:, 2]) * 1000.0
    phi = np.arctan2(xyz[:, 2], xyz[:, 1])
    u_index = np.floor(u_mm / float(axial_step_mm)).astype(np.int64)
    rho_index = np.floor(rho_mm / float(radial_step_mm)).astype(np.int64)
    bins = np.minimum(
        int(angle_bins) - 1,
        np.floor(phi / (0.5 * np.pi) * int(angle_bins)).astype(np.int64),
    )
    source_ids = (
        proposals["proposal_id"].astype(str).to_numpy()
        if "proposal_id" in proposals
        else proposals.index.astype(str).to_numpy()
    )
    result = pd.DataFrame(
        {
            "proposal_id": source_ids[finite],
            "x_m": xyz[:, 0], "y_m": xyz[:, 1], "z_m": xyz[:, 2],
            "u_mm": u_mm, "rho_mm": rho_mm, "phi_rad": phi,
            "u_index": u_index, "rho_index": rho_index, "angle_bin_id": bins,
        }
    )
    result["cell_id"] = [_cell_id(u, r) for u, r in zip(u_index, rho_index, strict=True)]
    result["proposal_beta_label_eligible"] = False
    result["proposal_beta_seed_eligible"] = False
    result["proposal_beta_branch_hint_eligible"] = False
    return result.sort_values(["cell_id", "angle_bin_id", "proposal_id"], kind="stable").reset_index(drop=True)


def _is_one_cell_hole(candidate: tuple[int, int], supported: set[tuple[int, int]]) -> bool:
    u, rho = candidate
    neighbors = {(u - 1, rho), (u + 1, rho), (u, rho - 1), (u, rho + 1)}
    # Four orthogonal supports is a deliberately conservative definition of an
    # isolated one-cell pocket.  It cannot turn a corridor or boundary gap into
    # a required cell.
    return neighbors.issubset(supported)


def register_required_shell_domain(
    quotient_proposals: pd.DataFrame,
    *,
    zero_xyz_m: Sequence[float],
    axial_step_mm: float = 15.0,
    radial_step_mm: float = 15.0,
    target_arc_step_mm: float = 10.0,
    support_distance_mm: float = 15.0,
    allow_one_cell_holes: bool = True,
    zero_core_cell_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Register zero-connected, proposal-supported ``(u,rho)`` shell cells.

    The second return value is a per-cell analytic ring registry.  A cell is a
    trajectory candidate only when every analytic target is genuinely close to
    a target-only proposal; a high coarse-bin count alone is insufficient.
    """

    if quotient_proposals.empty:
        return pd.DataFrame(columns=SHELL_CELL_COLUMNS), pd.DataFrame()
    zero = np.asarray(zero_xyz_m, dtype=float).reshape(3)
    rows: list[dict[str, Any]] = []
    proposal_xyz = _xyz(quotient_proposals)
    tree = cKDTree(proposal_xyz)
    grouped = quotient_proposals.groupby(["u_index", "rho_index"], sort=True)
    for (u, rho), group in grouped:
        u = int(u); rho = int(rho)
        u_center = (u + 0.5) * float(axial_step_mm)
        rho_center = (rho + 0.5) * float(radial_step_mm)
        n_intervals = max(4, int(np.ceil(np.pi * rho_center / (2.0 * float(target_arc_step_mm)))))
        phi = np.linspace(0.0, 0.5 * np.pi, n_intervals + 1)
        analytic = np.column_stack(
            [
                np.full(len(phi), zero[0] - u_center / 1000.0),
                rho_center / 1000.0 * np.cos(phi),
                rho_center / 1000.0 * np.sin(phi),
            ]
        )
        distance_mm, _ = tree.query(analytic, k=1)
        coverage = int(group["angle_bin_id"].nunique()) >= 6
        trajectory = bool(
            int(group["angle_bin_id"].nunique()) == 8
            and np.all(distance_mm * 1000.0 <= float(support_distance_mm) + 1.0e-12)
        )
        rows.append(
            {
                "cell_id": _cell_id(u, rho), "u_index": u, "rho_index": rho,
                "u_center_mm": u_center, "rho_center_mm": rho_center,
                "proposal_count": int(len(group)), "angle_bin_count": int(group["angle_bin_id"].nunique()),
                "coverage_ring_candidate": coverage, "trajectory_ring_candidate": trajectory,
                "required_shell_cell": False, "one_cell_hole": False,
                "n_phi_intervals": n_intervals,
                "analytic_support_p95_mm": float(np.percentile(distance_mm * 1000.0, 95)),
                "analytic_support_max_mm": float(np.max(distance_mm * 1000.0)),
            }
        )
    cells = pd.DataFrame.from_records(rows).sort_values("cell_id", kind="stable").reset_index(drop=True)
    supported = {(int(row.u_index), int(row.rho_index)) for row in cells.itertuples() if row.coverage_ring_candidate}
    if allow_one_cell_holes and supported:
        bounds = (min(u for u, _ in supported), max(u for u, _ in supported), min(r for _, r in supported), max(r for _, r in supported))
        holes = [
            (u, r) for u in range(bounds[0] + 1, bounds[1]) for r in range(bounds[2] + 1, bounds[3])
            if (u, r) not in supported and _is_one_cell_hole((u, r), supported)
        ]
        for u, rho in holes:
            hole_id = _cell_id(u, rho)
            existing = cells["cell_id"].eq(hole_id)
            if existing.any():
                # A low-angle-support proposal cell can itself be the isolated
                # pocket.  Promote that existing row instead of registering a
                # duplicate mesh cell with the same deterministic vertex IDs.
                cells.loc[existing, "coverage_ring_candidate"] = True
                cells.loc[existing, "one_cell_hole"] = True
            else:
                cells = pd.concat([cells, pd.DataFrame([{
                    "cell_id": hole_id, "u_index": u, "rho_index": rho,
                    "u_center_mm": (u + .5) * axial_step_mm, "rho_center_mm": (rho + .5) * radial_step_mm,
                    "proposal_count": 0, "angle_bin_count": 0, "coverage_ring_candidate": True,
                    "trajectory_ring_candidate": False, "required_shell_cell": False, "one_cell_hole": True,
                    "n_phi_intervals": max(4, int(np.ceil(np.pi * (rho + .5) * radial_step_mm / (2 * target_arc_step_mm)))),
                    "analytic_support_p95_mm": np.nan, "analytic_support_max_mm": np.nan,
                }])], ignore_index=True)
            supported.add((u, rho))
    # The runner passes snapped legacy/known-zero core cells when available.
    # A missing or disconnected explicit core is fail-closed; only the pure
    # target-only baseline falls back to the nearest proposal-supported cell.
    explicit_roots = {
        _as_cell_tuple(value) for value in (zero_core_cell_ids or ()) if _as_cell_tuple(value) in supported
    }
    if zero_core_cell_ids is not None:
        roots = sorted(explicit_roots)
    else:
        roots = [min(supported, key=lambda pair: (pair[0] ** 2 + pair[1] ** 2, pair))] if supported else []
    reachable: set[tuple[int, int]] = set()
    if roots:
        queue: deque[tuple[int, int]] = deque(roots)
        reachable.update(roots)
        while queue:
            u, rho = queue.popleft()
            for neighbor in ((u - 1, rho), (u + 1, rho), (u, rho - 1), (u, rho + 1)):
                if neighbor in supported and neighbor not in reachable:
                    reachable.add(neighbor); queue.append(neighbor)
    lookup = set(reachable)
    cells["required_shell_cell"] = [
        (int(row.u_index), int(row.rho_index)) in lookup for row in cells.itertuples()
    ]
    cells = cells.sort_values(["u_index", "rho_index"], kind="stable").reset_index(drop=True)
    rings = cells[cells["required_shell_cell"] & cells["trajectory_ring_candidate"]].copy()
    rings["ring_id"] = [stable_id("retry14_ring", cell) for cell in rings["cell_id"]]
    return cells, rings.reset_index(drop=True)


def build_shell_target_mesh(
    cells: pd.DataFrame,
    quotient_proposals: pd.DataFrame,
    *,
    zero_xyz_m: Sequence[float],
    maximum_solver_segment_mm: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Freeze mesh vertices, logical edges, and connector relays before solves."""

    zero = np.asarray(zero_xyz_m, dtype=float).reshape(3)
    tree = cKDTree(_xyz(quotient_proposals)) if len(quotient_proposals) else None
    vertices: list[dict[str, Any]] = []
    for cell in cells[cells["required_shell_cell"]].sort_values(["u_index", "rho_index"], kind="stable").itertuples():
        intervals = int(cell.n_phi_intervals)
        for index, phi in enumerate(np.linspace(0.0, .5 * np.pi, intervals + 1)):
            analytic = np.asarray([zero[0] - cell.u_center_mm / 1000.0, cell.rho_center_mm / 1000.0 * np.cos(phi), cell.rho_center_mm / 1000.0 * np.sin(phi)])
            support_mm = np.inf
            target = analytic
            kind = "analytic_trajectory" if bool(cell.trajectory_ring_candidate) else "coverage_proposal"
            if tree is not None:
                distance, position = tree.query(analytic, k=1)
                support_mm = float(distance * 1000.0)
                if kind == "coverage_proposal":
                    target = _xyz(quotient_proposals)[int(position)]
            vertex_id = stable_id("retry14_vertex", cell.cell_id, index, kind)
            vertices.append({
                "mesh_vertex_id": vertex_id, "cell_id": cell.cell_id, "ring_id": stable_id("retry14_ring", cell.cell_id),
                "phi_index": index, "phi_rad": float(phi), "x_m": target[0], "y_m": target[1], "z_m": target[2],
                "target_kind": kind, "proposal_support_distance_mm": support_mm,
                "label_role": "supervision", "label_status": "unresolved", "required_shell_cell": True,
                "trajectory_ring_candidate": bool(cell.trajectory_ring_candidate),
            })
    # Exact zero is a unique supervision node, never a ring member and never
    # part of a shell-cell denominator.  The runner supplies its known beta.
    zero_vertex_id = stable_id("retry14_vertex", "exact_zero")
    vertices.append({
        "mesh_vertex_id": zero_vertex_id, "cell_id": "exact_zero", "ring_id": None,
        "phi_index": -1, "phi_rad": 0.0, **dict(zip(XYZ_COLUMNS, zero, strict=True)),
        "target_kind": "exact_zero", "proposal_support_distance_mm": 0.0,
        "label_role": "supervision", "label_status": "seeded", "required_shell_cell": False,
        "trajectory_ring_candidate": False, "seam_class": "exact_zero",
    })
    vertex_frame = pd.DataFrame.from_records(vertices)
    by_cell = {cell: group.sort_values("phi_index", kind="stable") for cell, group in vertex_frame.groupby("cell_id", sort=True)}
    logical: list[dict[str, Any]] = []
    def add_edge(left: str, right: str, edge_type: str) -> None:
        if left == right:
            return
        pair = tuple(sorted((left, right)))
        logical.append({"logical_edge_id": stable_id("retry14_edge", edge_type, *pair), "left_mesh_vertex_id": left, "right_mesh_vertex_id": right, "mesh_edge_type": edge_type})
    # Angular edges stay within a cell.
    for group in by_cell.values():
        ids = group["mesh_vertex_id"].tolist()
        for left, right in zip(ids[:-1], ids[1:], strict=True): add_edge(left, right, "angular_edge")
    # Radial and axial nearest-normalised-angle matches.
    cell_rows = cells[cells["required_shell_cell"]].set_index("cell_id")
    for row in cell_rows.itertuples():
        for delta, edge_type in [((0, 1), "radial_edge"), ((1, 0), "axial_edge")]:
            neighbor = _cell_id(int(row.u_index) + delta[0], int(row.rho_index) + delta[1])
            if neighbor not in by_cell:
                continue
            left = by_cell[row.Index]; right = by_cell[neighbor]
            for candidate in left.itertuples():
                position = int(np.argmin(np.abs(right["phi_rad"].to_numpy(float) - float(candidate.phi_rad))))
                add_edge(str(candidate.mesh_vertex_id), str(right.iloc[position]["mesh_vertex_id"]), edge_type)
    # Register two symmetry-fixed zero corridors.  A generic phi metadata value
    # is not sufficient because coverage-proposal xyz may sit slightly off the
    # analytic plane; both endpoints and every interpolated relay must remain
    # in the corresponding fixed beta subspace.
    nonzero = vertex_frame[vertex_frame["mesh_vertex_id"].ne(zero_vertex_id)]
    for seam_name, seam_mask in (
        ("z_seam", nonzero["z_m"].abs().le(1.0e-12) & nonzero["y_m"].abs().gt(1.0e-12)),
        ("y_seam", nonzero["y_m"].abs().le(1.0e-12) & nonzero["z_m"].abs().gt(1.0e-12)),
    ):
        candidates = nonzero[seam_mask].copy()
        if candidates.empty:
            continue
        candidates["zero_distance_mm"] = np.linalg.norm(
            _xyz(candidates) - zero.reshape(1, 3), axis=1
        ) * 1000.0
        anchor = candidates.sort_values(
            ["zero_distance_mm", "mesh_vertex_id"], kind="stable"
        ).iloc[0]
        add_edge(zero_vertex_id, str(anchor["mesh_vertex_id"]), f"zero_{seam_name}_connector_edge")
    edges = pd.DataFrame.from_records(logical).drop_duplicates("logical_edge_id").sort_values("logical_edge_id", kind="stable").reset_index(drop=True)
    positions = vertex_frame.set_index("mesh_vertex_id")[list(XYZ_COLUMNS)]
    solver_segments: list[dict[str, Any]] = []
    for edge_position, edge in enumerate(edges.itertuples(index=False)):
        left = positions.loc[edge.left_mesh_vertex_id].to_numpy(float); right = positions.loc[edge.right_mesh_vertex_id].to_numpy(float)
        length = float(np.linalg.norm(right - left) * 1000.0)
        count = max(1, int(np.ceil(length / float(maximum_solver_segment_mm))))
        vertex_chain = [str(edge.left_mesh_vertex_id)]
        for step in range(1, count):
            xyz = left + (right - left) * step / count
            connector_id = stable_id("retry14_connector", edge.logical_edge_id, step)
            vertex_chain.append(connector_id)
        vertex_chain.append(str(edge.right_mesh_vertex_id))
        segment_ids: list[str] = []
        for segment_index, (left_id, right_id) in enumerate(zip(vertex_chain[:-1], vertex_chain[1:], strict=True)):
            left_xyz = left + (right - left) * segment_index / count
            right_xyz = left + (right - left) * (segment_index + 1) / count
            segment_id = stable_id("retry14_solver_segment", edge.logical_edge_id, segment_index)
            segment_ids.append(segment_id)
            solver_segments.append({
                "solver_segment_id": segment_id, "logical_edge_id": edge.logical_edge_id,
                "segment_index": segment_index, "left_mesh_vertex_id": left_id,
                "right_mesh_vertex_id": right_id, "left_x_m": left_xyz[0], "left_y_m": left_xyz[1], "left_z_m": left_xyz[2],
                "right_x_m": right_xyz[0], "right_y_m": right_xyz[1], "right_z_m": right_xyz[2],
                "segment_length_mm": float(np.linalg.norm(right_xyz - left_xyz) * 1000.0),
                "label_role": "connector_only" if (left_id not in positions.index or right_id not in positions.index) else "supervision",
            })
        # ``edges`` may contain hundreds of thousands of rows.  The stable
        # reset index above makes direct scalar assignment authoritative and
        # avoids four full-frame boolean scans per edge.
        edges.at[edge_position, "segment_chain_ids"] = "|".join(segment_ids)
        edges.at[edge_position, "segment_vertex_chain_ids"] = "|".join(vertex_chain)
        edges.at[edge_position, "logical_length_mm"] = length
        edges.at[edge_position, "mesh_edge_verified"] = False
    return vertex_frame, edges, pd.DataFrame.from_records(solver_segments)


def map_legacy_labels_to_mesh(
    legacy_labels: pd.DataFrame,
    vertices: pd.DataFrame,
    *,
    maximum_snap_mm: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map only zero-reachable legacy labels; mapping never marks a cell served."""

    if vertices.empty:
        return pd.DataFrame(), legacy_labels.copy(), vertices.copy()
    tree = cKDTree(_xyz(vertices))
    distance, location = tree.query(_xyz(legacy_labels), k=1)
    eligible = legacy_labels.copy()
    reachable = eligible.get("zero_reachable", pd.Series(True, index=eligible.index)).astype(bool).to_numpy()
    mask = (distance * 1000.0 <= float(maximum_snap_mm) + 1e-12) & reachable
    mappings: list[dict[str, Any]] = []
    for source, vertex_position, gap in zip(eligible[mask].to_dict("records"), location[mask], distance[mask] * 1000.0, strict=True):
        mappings.append({"legacy_label_id": str(source.get("physical_point_id", source.get("mesh_vertex_id"))), "mesh_vertex_id": str(vertices.iloc[int(vertex_position)]["mesh_vertex_id"]), "snap_distance_mm": float(gap), "mapping_role": "mesh_seed"})
    mapped_ids = {entry["legacy_label_id"] for entry in mappings}
    unused = eligible[~eligible.get("physical_point_id", pd.Series("", index=eligible.index)).astype(str).isin(mapped_ids)].copy()
    seeded = vertices.copy(); seeded["label_status"] = seeded["label_status"].where(~seeded["mesh_vertex_id"].isin({entry["mesh_vertex_id"] for entry in mappings}), "seeded")
    return pd.DataFrame.from_records(mappings), unused.reset_index(drop=True), seeded


def coverage_fill_required(metrics: Mapping[str, float], *, target_shell_coverage: float = .80, target_ring_fraction: float = .80, target_service_radius_mm: float = 20.0, allowed_gap_bins: int = 1) -> bool:
    return bool(metrics.get("shell_cell_coverage", 0.0) < target_shell_coverage or metrics.get("ring_completion_fraction", 0.0) < target_ring_fraction or metrics.get("proposal_service_p95_mm", np.inf) > target_service_radius_mm or metrics.get("maximum_angular_gap_bins", np.inf) > allowed_gap_bins)


def is_low_gain(before: Mapping[str, float], after: Mapping[str, float]) -> bool:
    return bool(after.get("shell_cell_coverage", 0.0) - before.get("shell_cell_coverage", 0.0) < .005 and after.get("ring_completion_fraction", 0.0) - before.get("ring_completion_fraction", 0.0) < .02 and after.get("complete_ring_count", 0.0) - before.get("complete_ring_count", 0.0) == 0.0)


def select_fill_targets(vertices: pd.DataFrame, *, maximum_targets: int) -> pd.DataFrame:
    """Return unresolved targets in the registered retry14 priority order."""

    frame = vertices[vertices["label_status"].isin(["unresolved", "local_hole"])].copy()
    for column, default in [("incomplete_trajectory_ring", False), ("angular_gap_bins", 0), ("unserved_required_shell_cell", False), ("proposal_service_distance_mm", 0.0), ("radial_or_axial_mesh_hole", False)]:
        if column not in frame: frame[column] = default
    return frame.sort_values(
        ["incomplete_trajectory_ring", "angular_gap_bins", "unserved_required_shell_cell", "proposal_service_distance_mm", "radial_or_axial_mesh_hole", "mesh_vertex_id"],
        ascending=[False, False, False, False, False, True], kind="stable",
    ).head(int(maximum_targets)).reset_index(drop=True)


def classify_candidate_endpoints(endpoints: pd.DataFrame) -> str:
    """Classify endpoints without averaging or choosing a branch silently."""

    accepted = endpoints[endpoints.get("success", pd.Series(False, index=endpoints.index)).astype(bool)].copy()
    if len(accepted) >= 2:
        beta = accepted.loc[:, BETA_COLUMNS].to_numpy(float)
        raw = np.max(np.abs(np.degrees(beta[:, None, :] - beta[None, :, :])), axis=2)
        if float(np.max(raw)) > 5.0 + 1e-12:
            return "branch_conflict"
        if np.all(accepted.get("fk_residual_mm", pd.Series(np.inf, index=accepted.index)).to_numpy(float) <= 8.0 + 1e-12):
            return "Gold"
    return "provisional_silver" if len(accepted) == 1 else "unresolved"


def choose_canonical_parent(endpoints: pd.DataFrame) -> pd.Series:
    """Tie-break a consistent multi-parent endpoint deterministically."""

    if classify_candidate_endpoints(endpoints) == "branch_conflict":
        raise ValueError("branch-conflict endpoints must not be aggregated")
    valid = endpoints[endpoints.get("success", pd.Series(False, index=endpoints.index)).astype(bool)].copy()
    if valid.empty: raise ValueError("no successful endpoint")
    for column, default in [("lineage_depth", np.inf), ("source_condition", np.inf), ("weighted_transition_deg", np.inf), ("parent_id", "")]:
        if column not in valid: valid[column] = default
    return valid.sort_values(["lineage_depth", "source_condition", "weighted_transition_deg", "parent_id"], kind="stable").iloc[0]


def audit_shell_mesh(vertices: pd.DataFrame, edges: pd.DataFrame, cells: pd.DataFrame) -> dict[str, float]:
    """Compute served-cell, complete-ring and logical-mesh cycle metrics."""

    node_ok = vertices.set_index("mesh_vertex_id")["label_status"].isin(["Gold", "certified_silver", "seeded"])
    verified = edges.get("mesh_edge_verified", pd.Series(False, index=edges.index)).astype(bool)
    served: set[str] = set()
    complete: set[str] = set()
    for cell in cells[cells["required_shell_cell"]].itertuples():
        subset = vertices[vertices["cell_id"].eq(cell.cell_id)]
        ids = set(subset["mesh_vertex_id"])
        angular = edges[edges.get("mesh_edge_type", pd.Series("", index=edges.index)).eq("angular_edge")]
        local_edges = angular[angular["left_mesh_vertex_id"].isin(ids) & angular["right_mesh_vertex_id"].isin(ids)]
        ordered = subset.sort_values("phi_index", kind="stable")["mesh_vertex_id"].astype(str).tolist()
        verified_pairs = {
            frozenset((str(edge.left_mesh_vertex_id), str(edge.right_mesh_vertex_id)))
            for edge in local_edges[verified.reindex(local_edges.index, fill_value=False)].itertuples()
        }
        complete_arc = bool(
            len(subset)
            and subset["mesh_vertex_id"].map(node_ok).all()
            and all(frozenset(pair) in verified_pairs for pair in zip(ordered[:-1], ordered[1:], strict=True))
        )
        if complete_arc: served.add(str(cell.cell_id))
        ring_id = stable_id("retry14_ring", cell.cell_id)
        if bool(cell.trajectory_ring_candidate) and complete_ring_status(vertices, edges, ring_id): complete.add(ring_id)
    good_edges = edges[verified].copy()
    ids = set(vertices["mesh_vertex_id"])
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in good_edges.itertuples():
        adjacency[str(edge.left_mesh_vertex_id)].add(str(edge.right_mesh_vertex_id)); adjacency[str(edge.right_mesh_vertex_id)].add(str(edge.left_mesh_vertex_id))
    components = 0; seen: set[str] = set()
    for node in ids:
        if node in seen: continue
        components += 1; queue = [node]; seen.add(node)
        while queue:
            current = queue.pop()
            for other in adjacency[current]:
                if other not in seen: seen.add(other); queue.append(other)
    required = int(cells["required_shell_cell"].sum())
    registered = int((cells["required_shell_cell"] & cells["trajectory_ring_candidate"]).sum())
    return {"shell_cell_coverage": len(served) / required if required else 0.0, "ring_completion_fraction": len(complete) / registered if registered else 0.0, "complete_ring_count": float(len(complete)), "mesh_cycle_rank": float(len(good_edges) - len(ids) + components), "served_required_shell_cell_count": float(len(served))}


def complete_ring_status(vertices: pd.DataFrame, edges: pd.DataFrame, ring_id: str) -> bool:
    """Return true only for a certified angular *cycle*, never a tree arc."""

    ring = vertices[vertices.get("ring_id", pd.Series("", index=vertices.index)).eq(str(ring_id))]
    if len(ring) < 3 or not ring.get("label_status", pd.Series("", index=ring.index)).isin(["Gold", "certified_silver", "seeded"]).all():
        return False
    ids = set(ring["mesh_vertex_id"].astype(str))
    angular = edges[
        edges.get("mesh_edge_type", pd.Series("", index=edges.index)).eq("angular_edge")
        & edges.get("mesh_edge_verified", pd.Series(False, index=edges.index)).astype(bool)
        & edges["left_mesh_vertex_id"].astype(str).isin(ids)
        & edges["right_mesh_vertex_id"].astype(str).isin(ids)
    ]
    if len(angular) != len(ids):
        return False
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in angular.itertuples():
        adjacency[str(edge.left_mesh_vertex_id)].add(str(edge.right_mesh_vertex_id))
        adjacency[str(edge.right_mesh_vertex_id)].add(str(edge.left_mesh_vertex_id))
    if set(adjacency) != ids or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return False
    seen = {next(iter(ids))}; queue = list(seen)
    while queue:
        current = queue.pop()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor); queue.append(neighbor)
    return seen == ids


def mesh_cycle_rank(vertices: pd.DataFrame, edges: pd.DataFrame) -> int:
    """Return the first Betti number of the verified logical mesh graph."""

    ids = set(vertices["mesh_vertex_id"].astype(str))
    good = edges[edges.get("mesh_edge_verified", pd.Series(False, index=edges.index)).astype(bool)]
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in good.itertuples():
        adjacency[str(edge.left_mesh_vertex_id)].add(str(edge.right_mesh_vertex_id))
        adjacency[str(edge.right_mesh_vertex_id)].add(str(edge.left_mesh_vertex_id))
    seen: set[str] = set(); components = 0
    for node in ids:
        if node in seen: continue
        components += 1; seen.add(node); queue = [node]
        while queue:
            current = queue.pop()
            for other in adjacency[current]:
                if other not in seen: seen.add(other); queue.append(other)
    return int(len(good) - len(ids) + components)


def reverse_delete_shell_dataset(nodes: pd.DataFrame, edges: pd.DataFrame, *, maximum_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Prune only non-mandatory leaves; ancestors/rings remain protected."""

    kept = nodes.copy().reset_index(drop=True); links = edges.copy().reset_index(drop=True); removed: list[str] = []
    protected = kept.get("mandatory", pd.Series(False, index=kept.index)).astype(bool) | kept.get("is_ring_member", pd.Series(False, index=kept.index)).astype(bool) | kept.get("is_lineage_ancestor", pd.Series(False, index=kept.index)).astype(bool)
    while len(kept) > int(maximum_rows):
        degree = pd.concat([links["left_mesh_vertex_id"], links["right_mesh_vertex_id"]]).value_counts() if len(links) else pd.Series(dtype=int)
        candidates = kept.loc[~protected & kept["mesh_vertex_id"].map(degree).fillna(0).le(1)].sort_values(["zero_radius_mm", "mesh_vertex_id"], ascending=[False, True], kind="stable")
        if candidates.empty: break
        victim = str(candidates.iloc[0]["mesh_vertex_id"]); removed.append(victim)
        kept = kept[~kept["mesh_vertex_id"].eq(victim)].reset_index(drop=True)
        links = links[~links["left_mesh_vertex_id"].eq(victim) & ~links["right_mesh_vertex_id"].eq(victim)].reset_index(drop=True)
        protected = kept.get("mandatory", pd.Series(False, index=kept.index)).astype(bool) | kept.get("is_ring_member", pd.Series(False, index=kept.index)).astype(bool) | kept.get("is_lineage_ancestor", pd.Series(False, index=kept.index)).astype(bool)
    return kept, links, removed


def mirror_nodes_and_mesh_edges(nodes: pd.DataFrame, edges: pd.DataFrame, *, zero_vertex_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror all node and mesh-edge classes under G4, merging exact zero."""

    member: dict[tuple[str, str], str] = {}
    rows: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()
    for node in nodes.to_dict("records"):
        xyz = np.asarray([node[name] for name in XYZ_COLUMNS], dtype=float)
        beta = np.asarray([node[name] for name in BETA_COLUMNS], dtype=float) if all(name in node for name in BETA_COLUMNS) else None
        y_fixed = bool(abs(float(xyz[1])) <= 1.0e-12)
        z_fixed = bool(abs(float(xyz[2])) <= 1.0e-12)
        beta_y_fixed = bool(beta is not None and np.max(np.abs(beta[[0, 2, 4]])) <= 1.0e-8)
        beta_z_fixed = bool(beta is not None and np.max(np.abs(beta[[1, 3, 5]])) <= 1.0e-8)
        symmetry_fixed_label = bool((not y_fixed or beta_y_fixed) and (not z_fixed or beta_z_fixed))
        for element in SYMMETRY_BETA_SIGNS:
            transformed_xyz = transform_xyz(xyz, element)
            transformed_xyz[np.abs(transformed_xyz) <= 1.0e-12] = 0.0
            # A symmetry-fixed seam endpoint is the same physical node under
            # two group elements.  Merge by transformed task coordinate so
            # four mirrored quotient arcs share seams and close into a cycle.
            coordinate_key = tuple(np.round(transformed_xyz, 12).tolist())
            if str(node["mesh_vertex_id"]) == str(zero_vertex_id):
                target_id = str(zero_vertex_id)
            elif symmetry_fixed_label:
                target_id = stable_id("retry14_full_vertex", node["mesh_vertex_id"], *coordinate_key)
            else:
                # Non-equivariant lineage-only seam evidence represents
                # distinct branch states at the same xyz.  It must never be
                # silently merged into a canonical seam supervision node.
                target_id = stable_id(
                    "retry14_full_vertex", node["mesh_vertex_id"], *coordinate_key, element
                )
            member[(str(node["mesh_vertex_id"]), element)] = target_id
            if target_id in seen_node_ids:
                continue
            seen_node_ids.add(target_id)
            transformed = {**node, **dict(zip(XYZ_COLUMNS, transformed_xyz, strict=True)), "mesh_vertex_id": target_id, "symmetry_element": element, "fundamental_mesh_vertex_id": str(node["mesh_vertex_id"])}
            if beta is not None:
                transformed_beta = transform_beta(beta, element); transformed.update(dict(zip(BETA_COLUMNS, transformed_beta, strict=True))); transformed.update(dict(zip((f"theta{i}_rad" for i in range(1, 31)), beta_to_theta(transformed_beta), strict=True)))
            rows.append(transformed)
    full_nodes = pd.DataFrame.from_records(rows).drop_duplicates("mesh_vertex_id", keep="first").reset_index(drop=True)
    full_edges: list[dict[str, Any]] = []
    for edge in edges.to_dict("records"):
        for element in SYMMETRY_BETA_SIGNS:
            left = member[(str(edge["left_mesh_vertex_id"]), element)]; right = member[(str(edge["right_mesh_vertex_id"]), element)]
            if left == right: continue
            full_edges.append({**edge, "full_logical_edge_id": stable_id("retry14_full_edge", edge["logical_edge_id"], element), "left_mesh_vertex_id": left, "right_mesh_vertex_id": right, "symmetry_element": element})
    result = pd.DataFrame.from_records(full_edges)
    if len(result):
        pair = result.apply(lambda row: "|".join(sorted((str(row.left_mesh_vertex_id), str(row.right_mesh_vertex_id)))) + ":" + str(row.mesh_edge_type), axis=1)
        result = result.loc[~pair.duplicated()].reset_index(drop=True)
    return full_nodes, result


def connector_loss_mask(nodes: pd.DataFrame) -> pd.Series:
    return ~nodes.get("label_role", pd.Series("supervision", index=nodes.index)).eq("connector_only")


def assert_holdout_macroblock_isolation(nodes: pd.DataFrame) -> None:
    roles = nodes.groupby("quotient_macroblock_id", sort=False)["split_role"].agg(lambda values: set(map(str, values)))
    leaking = {block: values for block, values in roles.items() if "trajectory_holdout" in values and len(values) != 1}
    if leaking: raise ValueError(f"trajectory holdout macroblock leakage: {sorted(leaking)}")


# Runner-facing names keep the stage protocol readable while retaining the
# explicit longer names above for direct unit tests and audit notebooks.
register_shell_domain = register_required_shell_domain
build_target_mesh = build_shell_target_mesh
expand_mesh_symmetry = mirror_nodes_and_mesh_edges
select_dataset = reverse_delete_shell_dataset


__all__ = [
    "audit_shell_mesh", "assert_holdout_macroblock_isolation", "build_shell_target_mesh", "build_target_mesh", "choose_canonical_parent",
    "classify_candidate_endpoints", "complete_ring_status", "connector_loss_mask", "coverage_fill_required", "expand_mesh_symmetry", "is_low_gain",
    "map_legacy_labels_to_mesh", "mesh_cycle_rank", "mirror_nodes_and_mesh_edges", "quotient_cylindrical_proposals",
    "register_required_shell_domain", "register_shell_domain", "reverse_delete_shell_dataset", "select_dataset", "select_fill_targets",
]
