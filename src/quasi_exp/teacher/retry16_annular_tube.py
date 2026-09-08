"""Retry16 annular/tubular task-domain helpers.

Retry15 assumed that every registered cross-section was a solid disk rooted at
``rho=0``.  The observed two-pool workspace proxy is instead shell-like away
from exact zero.  This module keeps the exact-zero point as a separate root,
extracts a full-sector annulus at each axial bin, and joins the two by one
explicit connector path.  Connector targets never become annular coverage
units.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from quasi_exp.model.sampling import beta_to_theta

from .retry12_symmetry import (
    BETA_COLUMNS,
    SYMMETRY_BETA_SIGNS,
    THETA_COLUMNS,
    XYZ_COLUMNS,
    stable_id,
    transform_beta,
    transform_xyz,
)
from .retry15_canonical_graph import Omega600Contract


@dataclass(frozen=True)
class AnnularPolicy:
    axial_step_mm: float = 10.0
    radial_step_mm: float = 10.0
    full_sector_count: int = 16
    inner_margin_mm: float = 10.0
    outer_margin_mm: float = 10.0
    slope_limit_mm: float = 20.0
    minimum_thickness_mm: float = 20.0
    root_link_maximum_mm: float = 25.0

    def __post_init__(self) -> None:
        if self.axial_step_mm <= 0 or self.radial_step_mm <= 0:
            raise ValueError("annular steps must be positive")
        if self.full_sector_count < 4 or self.full_sector_count % 4:
            raise ValueError("annular sector count must be a multiple of four")
        if min(self.inner_margin_mm, self.outer_margin_mm) < 0:
            raise ValueError("annular margins must be non-negative")
        if min(self.slope_limit_mm, self.minimum_thickness_mm, self.root_link_maximum_mm) <= 0:
            raise ValueError("annular slope, thickness and root link must be positive")


def _robust_keys(support: pd.DataFrame) -> set[tuple[int, int, int]]:
    required = {"u_index", "rho_index", "sector_index", "robust_supported"}
    if not required <= set(support):
        raise ValueError("workspace support is missing annular columns")
    return {
        tuple(map(int, row))
        for row in support.loc[
            support["robust_supported"].astype(bool),
            ["u_index", "rho_index", "sector_index"],
        ].itertuples(index=False, name=None)
    }


def intersect_support(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return the exact robust-voxel intersection of two nested probe scales."""

    keys = _robust_keys(left) & _robust_keys(right)
    rows = []
    for u_index, rho_index, sector_index in sorted(keys):
        rows.append(
            {
                "u_index": u_index,
                "rho_index": rho_index,
                "sector_index": sector_index,
                "robust_supported": True,
            }
        )
    return pd.DataFrame(rows, columns=["u_index", "rho_index", "sector_index", "robust_supported"])


def _longest_run(values: Sequence[int]) -> tuple[int, int] | None:
    ordered = sorted(set(map(int, values)))
    if not ordered:
        return None
    runs: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    runs.append((start, previous))
    return max(runs, key=lambda item: (item[1] - item[0] + 1, item[1], -item[0]))


def raw_annular_profile(
    support: pd.DataFrame,
    *,
    policy: AnnularPolicy | None = None,
    profile_id: str,
) -> pd.DataFrame:
    """Extract the longest radial interval supported in every angular sector."""

    options = AnnularPolicy() if policy is None else policy
    keys = _robust_keys(support)
    maximum_u = int(round(Omega600Contract().length_mm / options.axial_step_mm))
    by_u_sector: dict[tuple[int, int], set[int]] = {}
    for u_index, rho_index, sector_index in keys:
        by_u_sector.setdefault((u_index, sector_index), set()).add(rho_index)
    rows: list[dict[str, Any]] = []
    for u_index in range(maximum_u):
        common: set[int] | None = None
        for sector_index in range(options.full_sector_count):
            occupied = by_u_sector.get((u_index, sector_index), set())
            common = set(occupied) if common is None else common & occupied
        run = _longest_run(() if common is None else common)
        if run is None:
            inner = outer = math.nan
            thickness = 0.0
            supported = False
        else:
            inner = run[0] * options.radial_step_mm + options.inner_margin_mm
            outer = (run[1] + 1) * options.radial_step_mm - options.outer_margin_mm
            thickness = max(0.0, outer - inner)
            supported = bool(thickness >= options.minimum_thickness_mm)
        rows.append(
            {
                "u_index": u_index,
                "u_center_mm": (u_index + 0.5) * options.axial_step_mm,
                "raw_inner_radius_mm": inner,
                "raw_outer_radius_mm": outer,
                "raw_thickness_mm": thickness,
                "all_sector_annulus_supported": supported,
                "profile_id": profile_id,
            }
        )
    return pd.DataFrame(rows)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    positions = np.flatnonzero(np.asarray(mask, dtype=bool))
    return _longest_run(positions)


def _lipschitz_majorant(values: np.ndarray, *, limit_mm: float) -> np.ndarray:
    return np.asarray(
        [max(float(values[j]) - limit_mm * abs(i - j) for j in range(len(values))) for i in range(len(values))],
        dtype=float,
    )


def _lipschitz_minorant(values: np.ndarray, *, limit_mm: float) -> np.ndarray:
    return np.asarray(
        [min(float(values[j]) + limit_mm * abs(i - j) for j in range(len(values))) for i in range(len(values))],
        dtype=float,
    )


def smooth_annular_profile(
    raw: pd.DataFrame,
    *,
    policy: AnnularPolicy | None = None,
    profile_id: str,
    domain_class: str,
) -> pd.DataFrame:
    """Create the largest contiguous Lipschitz annulus contained in ``raw``."""

    options = AnnularPolicy() if policy is None else policy
    required = {
        "u_index",
        "u_center_mm",
        "raw_inner_radius_mm",
        "raw_outer_radius_mm",
        "all_sector_annulus_supported",
    }
    if not required <= set(raw):
        raise ValueError("raw annular profile is incomplete")
    ordered = raw.sort_values("u_index", kind="stable").reset_index(drop=True)
    run = _longest_true_run(ordered["all_sector_annulus_supported"].to_numpy(bool))
    if run is None:
        raise ValueError("no full-sector annular run exists")
    start, stop = run
    segment = ordered.iloc[start : stop + 1].copy()
    inner = _lipschitz_majorant(
        segment["raw_inner_radius_mm"].to_numpy(float),
        limit_mm=options.slope_limit_mm,
    )
    outer = _lipschitz_minorant(
        segment["raw_outer_radius_mm"].to_numpy(float),
        limit_mm=options.slope_limit_mm,
    )
    feasible = outer - inner >= options.minimum_thickness_mm - 1.0e-9
    feasible_run = _longest_true_run(feasible)
    if feasible_run is None:
        raise ValueError("annular slope constraints remove every supported interval")
    local_start, local_stop = feasible_run
    segment = segment.iloc[local_start : local_stop + 1].copy().reset_index(drop=True)
    inner = inner[local_start : local_stop + 1]
    outer = outer[local_start : local_stop + 1]
    # Re-run after trimming so excluded endpoints cannot constrain the result.
    inner = _lipschitz_majorant(inner, limit_mm=options.slope_limit_mm)
    outer = _lipschitz_minorant(outer, limit_mm=options.slope_limit_mm)
    if np.any(outer - inner < options.minimum_thickness_mm - 1.0e-9):
        raise ValueError("annular profile is thinner than the registered minimum")
    result = segment.loc[:, ["u_index", "u_center_mm"]].copy()
    result["inner_radius_mm"] = inner
    result["outer_radius_mm"] = outer
    result["thickness_mm"] = outer - inner
    result["domain_class"] = str(domain_class)
    result["profile_id"] = str(profile_id)
    return result


def build_nested_annular_profiles(
    final_support: pd.DataFrame,
    comparison_support: pd.DataFrame,
    *,
    policy: AnnularPolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build final-scale expanded and two-scale-intersection core profiles."""

    options = AnnularPolicy() if policy is None else policy
    raw_expanded = raw_annular_profile(
        final_support,
        policy=options,
        profile_id="retry16_raw_expanded_annulus",
    )
    raw_core = raw_annular_profile(
        intersect_support(final_support, comparison_support),
        policy=options,
        profile_id="retry16_raw_stable_core_annulus",
    )
    expanded = smooth_annular_profile(
        raw_expanded,
        policy=options,
        profile_id="retry16_expanded_annulus",
        domain_class="expanded",
    )
    core = smooth_annular_profile(
        raw_core,
        policy=options,
        profile_id="retry16_stable_core_annulus",
        domain_class="stable_core",
    )
    return raw_expanded, raw_core, expanded, core


def annular_volume_mm3(profile: pd.DataFrame, *, axial_step_mm: float = 10.0) -> float:
    inner = profile["inner_radius_mm"].to_numpy(float)
    outer = profile["outer_radius_mm"].to_numpy(float)
    return float(np.sum(0.25 * np.pi * (np.square(outer) - np.square(inner)) * axial_step_mm))


def _profile_lookup(profile: pd.DataFrame) -> dict[int, tuple[float, float]]:
    return {
        int(row.u_index): (float(row.inner_radius_mm), float(row.outer_radius_mm))
        for row in profile.itertuples(index=False)
    }


def _in_annulus(profile: pd.DataFrame, *, u_index: int, rho_mm: float) -> bool:
    bounds = _profile_lookup(profile).get(int(u_index))
    return bool(bounds is not None and bounds[0] - 1.0e-9 <= rho_mm <= bounds[1] + 1.0e-9)


def root_connector_voxels(
    support: pd.DataFrame,
    expanded_profile: pd.DataFrame,
    *,
    policy: AnnularPolicy | None = None,
) -> pd.DataFrame:
    """Find one robust six-neighbour path from exact zero into the annulus."""

    options = AnnularPolicy() if policy is None else policy
    keys = _robust_keys(support)
    quotient_keys = {key for key in keys if 0 <= key[2] < options.full_sector_count // 4}
    if not quotient_keys:
        raise ValueError("no robust quotient voxel can connect exact zero")
    phase_step = 2.0 * np.pi / options.full_sector_count

    def xyz_mm(key: tuple[int, int, int]) -> np.ndarray:
        u, rho, sector = key
        radius = (rho + 0.5) * options.radial_step_mm
        phase = (sector + 0.5) * phase_step
        return np.asarray(
            [
                (u + 0.5) * options.axial_step_mm,
                radius * math.cos(phase),
                radius * math.sin(phase),
            ],
            dtype=float,
        )

    start = min(
        quotient_keys,
        key=lambda key: (float(np.linalg.norm(xyz_mm(key))), key),
    )
    if float(np.linalg.norm(xyz_mm(start))) > options.root_link_maximum_mm + 1.0e-9:
        raise ValueError("nearest robust voxel is too far from exact zero")
    goals = {
        key
        for key in quotient_keys
        if _in_annulus(
            expanded_profile,
            u_index=key[0],
            rho_mm=(key[1] + 0.5) * options.radial_step_mm,
        )
    }
    if not goals:
        raise ValueError("expanded annulus has no robust connector goal")
    queue: deque[tuple[int, int, int]] = deque([start])
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    goal: tuple[int, int, int] | None = None
    while queue:
        current = queue.popleft()
        if current in goals:
            goal = current
            break
        u, rho, sector = current
        neighbours = (
            (u - 1, rho, sector),
            (u + 1, rho, sector),
            (u, rho - 1, sector),
            (u, rho + 1, sector),
            (u, rho, sector - 1),
            (u, rho, sector + 1),
        )
        for neighbour in neighbours:
            if neighbour not in quotient_keys or neighbour in parent:
                continue
            parent[neighbour] = current
            queue.append(neighbour)
    if goal is None:
        raise ValueError("robust support has no root-to-annulus connector")
    path = []
    current: tuple[int, int, int] | None = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    rows = []
    for ordinal, key in enumerate(path):
        u_mm, y_mm, z_mm = xyz_mm(key)
        rows.append(
            {
                "connector_ordinal": ordinal,
                "u_index": key[0],
                "rho_index": key[1],
                "sector_index": key[2],
                "u_mm": u_mm,
                "rho_mm": math.hypot(y_mm, z_mm),
                "y_mm": y_mm,
                "z_mm": z_mm,
                "robust_voxel_id": stable_id("retry16_root_voxel", *key),
                "annular_coverage_eligible": False,
            }
        )
    return pd.DataFrame(rows)


def _stable_xyz_key(xyz: Sequence[float]) -> tuple[int, int, int]:
    return tuple(np.rint(np.asarray(xyz, dtype=float) * 1.0e9).astype(np.int64).tolist())


def _circle_levels(core: pd.DataFrame, *, separation_bins: int = 10) -> tuple[int, int, int]:
    eligible = core.loc[
        core["outer_radius_mm"].ge(100.0), "u_index"
    ].astype(int).tolist()
    best: tuple[float, tuple[int, int, int]] | None = None
    lookup = core.set_index("u_index")
    for left in eligible:
        for middle in eligible:
            if middle - left < separation_bins:
                continue
            for right in eligible:
                if right - middle < separation_bins:
                    continue
                indices = (left, middle, right)
                widths = [float(lookup.loc[index, "thickness_mm"]) for index in indices]
                score = (min(widths), right - left, -left)
                if best is None or score > (best[0], best[1][2] - best[1][0], -best[1][0]):
                    best = (score[0], indices)
    if best is None:
        raise ValueError("stable annulus cannot register three separated circle levels")
    return best[1]


def _lattice_u_indices(profile: pd.DataFrame, spacing_mm: float, axial_step_mm: float) -> list[int]:
    available = profile["u_index"].astype(int).tolist()
    stride = max(1, int(round(spacing_mm / axial_step_mm)))
    selected = available[::stride]
    if available[-1] not in selected:
        selected.append(available[-1])
    return selected


def build_annular_target_registry(
    expanded_profile: pd.DataFrame,
    stable_core_profile: pd.DataFrame,
    connector: pd.DataFrame,
    proposal_xyz_m: np.ndarray,
    *,
    target_budget: int,
    maximum_circle_step_mm: float = 10.0,
    spacing_minimum_mm: float = 20.0,
    spacing_maximum_mm: float = 40.0,
    contract: Omega600Contract | None = None,
    policy: AnnularPolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    """Freeze a budgeted quotient target denominator inside the annular tube."""

    domain = Omega600Contract() if contract is None else contract
    options = AnnularPolicy() if policy is None else policy
    proposals = np.asarray(proposal_xyz_m, dtype=float).reshape(-1, 3)
    if not np.isfinite(proposals).all():
        raise ValueError("proposal xyz must be finite")
    expanded = _profile_lookup(expanded_profile)
    core = _profile_lookup(stable_core_profile)
    circle_levels = _circle_levels(stable_core_profile)

    def build(spacing_mm: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        records: dict[tuple[int, int, int], dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        circle_rows: list[dict[str, Any]] = []

        def add(
            xyz: Sequence[float],
            *,
            role: str,
            mandatory: bool,
            coverage_eligible: bool,
            domain_class: str,
            u_index: int,
            circle_id: str = "",
        ) -> str:
            point = np.asarray(xyz, dtype=float).reshape(3)
            key = _stable_xyz_key(point)
            target_id = stable_id("retry16_target", *key)
            rho_mm = float(1000.0 * np.hypot(point[1], point[2]))
            row = {
                "target_id": target_id,
                "x_m": float(point[0]),
                "y_m": float(point[1]),
                "z_m": float(point[2]),
                "u_mm": float(1000.0 * (domain.x0_m - point[0])),
                "rho_mm": rho_mm,
                "u_index": int(u_index),
                "target_role": str(role),
                "mandatory": bool(mandatory),
                "annular_coverage_eligible": bool(coverage_eligible),
                "domain_class": str(domain_class),
                "circle_id": str(circle_id),
                "proposal_beta_label_eligible": False,
                "proposal_beta_seed_eligible": False,
                "proposal_beta_warm_start_eligible": False,
                "proposal_beta_branch_hint_eligible": False,
            }
            previous = records.get(key)
            if previous is None or (mandatory and not previous["mandatory"]):
                records[key] = row
            elif circle_id and not previous["circle_id"]:
                previous.update(row)
            return target_id

        zero_id = add(
            (domain.x0_m, 0.0, 0.0),
            role="exact_zero",
            mandatory=True,
            coverage_eligible=False,
            domain_class="root",
            u_index=0,
        )
        previous_id = zero_id
        for row in connector.sort_values("connector_ordinal", kind="stable").itertuples(index=False):
            target_id = add(
                (
                    domain.x0_m - float(row.u_mm) / 1000.0,
                    float(row.y_mm) / 1000.0,
                    float(row.z_mm) / 1000.0,
                ),
                role="root_connector",
                mandatory=True,
                coverage_eligible=False,
                domain_class="connector",
                u_index=int(row.u_index),
            )
            pair = tuple(sorted((previous_id, target_id)))
            edges[(pair[0], pair[1], "root_connector")] = {
                "edge_id": stable_id("retry16_root_edge", *pair),
                "left_target_id": pair[0],
                "right_target_id": pair[1],
                "edge_type": "root_connector",
                "circle_id": "",
                "mandatory": True,
            }
            previous_id = target_id

        # The inverse-section denominator is frozen only after choosing this
        # spacing.  Candidate lattice points are not a pre-existing required
        # mesh and are never truncated after registration.
        for u_index in _lattice_u_indices(expanded_profile, spacing_mm, options.axial_step_mm):
            inner, outer = expanded[u_index]
            u_mm = (u_index + 0.5) * options.axial_step_mm
            coordinates = np.arange(0.0, outer + 1.0e-9, spacing_mm)
            for y_mm in coordinates:
                for z_mm in coordinates:
                    rho_mm = math.hypot(float(y_mm), float(z_mm))
                    if rho_mm < inner - 1.0e-9 or rho_mm > outer + 1.0e-9:
                        continue
                    domain_class = "stable_core" if (
                        u_index in core and core[u_index][0] - 1.0e-9 <= rho_mm <= core[u_index][1] + 1.0e-9
                    ) else "expanded"
                    role = "seam" if y_mm == 0.0 or z_mm == 0.0 else "annular_interior"
                    add(
                        (domain.x0_m - u_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0),
                        role=role,
                        mandatory=role == "seam",
                        coverage_eligible=True,
                        domain_class=domain_class,
                        u_index=u_index,
                    )

        core_lookup = stable_core_profile.set_index("u_index")
        for level_rank, u_index in enumerate(circle_levels):
            row = core_lookup.loc[u_index]
            inner = float(row["inner_radius_mm"])
            outer = float(row["outer_radius_mm"])
            u_mm = float(row["u_center_mm"])
            for radial_rank, fraction in enumerate((0.2, 0.5, 0.8)):
                radius_mm = inner + fraction * (outer - inner)
                circle_id = f"retry16_circle_u{u_index:03d}_r{radial_rank}"
                full_segments = max(16, int(math.ceil(2.0 * math.pi * radius_mm / maximum_circle_step_mm)))
                full_segments += (-full_segments) % 4
                phases = np.linspace(0.0, 0.5 * math.pi, full_segments // 4 + 1)
                ids = []
                for phase_index, phase in enumerate(phases):
                    target_id = add(
                        (
                            domain.x0_m - u_mm / 1000.0,
                            radius_mm * math.cos(float(phase)) / 1000.0,
                            radius_mm * math.sin(float(phase)) / 1000.0,
                        ),
                        role="heldout_circle",
                        mandatory=True,
                        coverage_eligible=True,
                        domain_class="stable_core",
                        u_index=u_index,
                        circle_id=circle_id,
                    )
                    ids.append(target_id)
                    circle_rows.append(
                        {
                            "circle_id": circle_id,
                            "level_rank": level_rank,
                            "radial_rank": radial_rank,
                            "u_index": u_index,
                            "u_mm": u_mm,
                            "radius_mm": radius_mm,
                            "diameter_mm": 2.0 * radius_mm,
                            "phase_index": phase_index,
                            "full_segment_count": full_segments,
                            "target_id": target_id,
                            "held_out": True,
                        }
                    )
                for left_id, right_id in zip(ids[:-1], ids[1:], strict=True):
                    pair = tuple(sorted((left_id, right_id)))
                    edges[(pair[0], pair[1], "circle")] = {
                        "edge_id": stable_id("retry16_circle_edge", circle_id, *pair),
                        "left_target_id": pair[0],
                        "right_target_id": pair[1],
                        "edge_type": "circle",
                        "circle_id": circle_id,
                        "mandatory": True,
                    }

        registry = pd.DataFrame(records.values()).sort_values(
            ["mandatory", "u_index", "rho_mm", "target_id"],
            ascending=[False, True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        edge_frame = pd.DataFrame(edges.values())
        circle_frame = pd.DataFrame(circle_rows)
        return registry, edge_frame, circle_frame

    selected: tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    for spacing_mm in np.arange(spacing_minimum_mm, spacing_maximum_mm + 1.0e-9, 1.0):
        registry, edges, circles = build(float(spacing_mm))
        if len(registry) <= int(target_budget):
            selected = float(spacing_mm), registry, edges, circles
            break
    if selected is None:
        raise ValueError("annular target registry exceeds the registered budget")
    spacing_mm, registry, edges, circles = selected
    tree = cKDTree(proposals)
    distance_m, nearest = tree.query(registry.loc[:, XYZ_COLUMNS].to_numpy(float), k=1)
    registry["proposal_service_mm"] = np.asarray(distance_m, dtype=float) * 1000.0
    registry["proposal_service_pass"] = registry["proposal_service_mm"].le(spacing_mm + 1.0e-9)
    registry["nearest_proposal_index"] = np.asarray(nearest, dtype=np.int64)
    registry["required_voxel_id"] = registry["target_id"]
    registry["target_ordinal"] = np.arange(len(registry), dtype=np.int64)
    coverage = registry[registry["annular_coverage_eligible"].astype(bool)]
    outside = []
    for row in coverage.itertuples(index=False):
        outside.append(not _in_annulus(expanded_profile, u_index=int(row.u_index), rho_mm=float(row.rho_mm)))
    registry["outside_expanded_annulus"] = False
    registry.loc[coverage.index, "outside_expanded_annulus"] = outside
    summary = {
        "target_count": int(len(registry)),
        "coverage_target_count": int(len(coverage)),
        "stable_core_target_count": int(registry["domain_class"].eq("stable_core").sum()),
        "target_budget": int(target_budget),
        "target_spacing_mm": spacing_mm,
        "budget_feasible": bool(len(registry) <= int(target_budget)),
        "mandatory_target_count": int(registry["mandatory"].sum()),
        "connector_target_count": int(registry["target_role"].eq("root_connector").sum()),
        "registered_circle_count": int(circles["circle_id"].nunique()),
        "large_circle_count": int(circles.loc[circles["radius_mm"].ge(100.0), "circle_id"].nunique()),
        "minimum_circle_radius_mm": float(circles["radius_mm"].min()),
        "maximum_circle_radius_mm": float(circles["radius_mm"].max()),
        "proposal_service_pass_fraction": float(registry["proposal_service_pass"].mean()),
        "outside_expanded_annulus_count": int(registry["outside_expanded_annulus"].sum()),
        "exact_zero_count": int(registry["target_role"].eq("exact_zero").sum()),
        "axis_core_target_count": int(registry["target_role"].eq("axis_core").sum()),
        "expanded_first_u_mm": float(expanded_profile["u_center_mm"].min()),
        "expanded_last_u_mm": float(expanded_profile["u_center_mm"].max()),
        "stable_core_first_u_mm": float(stable_core_profile["u_center_mm"].min()),
        "stable_core_last_u_mm": float(stable_core_profile["u_center_mm"].max()),
    }
    return registry, edges, circles, summary


def deterministic_target_sample(targets: pd.DataFrame, *, limit: int, salt: str) -> pd.DataFrame:
    mandatory = targets[targets["mandatory"].astype(bool)].copy()
    if len(mandatory) > int(limit):
        raise ValueError("mandatory annular targets exceed the registered sample limit")
    remaining = targets[~targets["target_id"].isin(mandatory["target_id"])].copy()
    remaining["_hash"] = remaining["target_id"].astype(str).map(
        lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    )
    selected = remaining.sort_values(
        ["domain_class", "_hash", "u_index", "rho_mm", "target_id"], kind="stable"
    ).head(max(0, int(limit) - len(mandatory))).drop(columns="_hash")
    return pd.concat([mandatory, selected], ignore_index=True, sort=False).sort_values(
        ["u_index", "rho_mm", "target_id"], kind="stable"
    ).reset_index(drop=True)


def expand_annular_labels(
    labels: pd.DataFrame,
    targets: pd.DataFrame,
    environment: Any,
    *,
    residual_maximum_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand quotient labels without duplicating fixed nonzero axis points."""

    target_columns = [
        "target_id",
        *XYZ_COLUMNS,
        "target_role",
        "domain_class",
        "circle_id",
        "annular_coverage_eligible",
    ]
    merged = labels.drop(columns=[name for name in XYZ_COLUMNS if name in labels], errors="ignore").merge(
        targets.loc[:, target_columns], on="target_id", how="inner", validate="one_to_one"
    )
    expanded: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    for row in merged.to_dict("records"):
        xyz = np.asarray([row[name] for name in XYZ_COLUMNS], dtype=float)
        beta = np.asarray([row[name] for name in BETA_COLUMNS], dtype=float)
        y_zero = abs(float(xyz[1])) <= 1.0e-12
        z_zero = abs(float(xyz[2])) <= 1.0e-12
        if str(row["target_role"]) == "exact_zero":
            elements = ("identity",)
        elif y_zero and z_zero:
            elements = ("identity",)
        elif y_zero:
            elements = ("identity", "mirror_z")
        elif z_zero:
            elements = ("identity", "mirror_y")
        else:
            elements = tuple(SYMMETRY_BETA_SIGNS)
        orbit_id = stable_id("retry16_orbit", row["target_id"])
        members = []
        valid = True
        for element in elements:
            member_xyz = transform_xyz(xyz, element)
            member_beta = transform_beta(beta, element)
            achieved = np.asarray(environment.fk(member_beta.reshape(1, 6)), dtype=float).reshape(3)
            residual_mm = float(np.linalg.norm(achieved - member_xyz) * 1000.0)
            valid = valid and bool(
                np.isfinite(member_beta).all()
                and np.all(member_beta >= bounds[:, 0] - 1.0e-12)
                and np.all(member_beta <= bounds[:, 1] + 1.0e-12)
                and residual_mm <= residual_maximum_mm + 1.0e-12
            )
            members.append(
                {
                    **row,
                    **dict(zip(XYZ_COLUMNS, member_xyz, strict=True)),
                    **dict(zip(BETA_COLUMNS, member_beta, strict=True)),
                    **dict(zip(THETA_COLUMNS, beta_to_theta(member_beta), strict=True)),
                    "symmetry_orbit_id": orbit_id,
                    "symmetry_element": element,
                    "orbit_size": len(elements),
                    "fk_residual_mm": residual_mm,
                    "proposal_beta_used": False,
                }
            )
        if valid:
            expanded.extend(members)
        else:
            rejected.append({**row, "orbit_rejection_reason": "invalid_orbit_member"})
    result = pd.DataFrame(expanded)
    if len(result) and result.duplicated(list(XYZ_COLUMNS)).any():
        duplicates = result[result.duplicated(list(XYZ_COLUMNS), keep=False)]
        # Seams may meet numerically at signed zero.  They must carry exactly
        # the same beta or the expansion is not a single-valued dataset.
        for _xyz, frame in duplicates.groupby(list(XYZ_COLUMNS), dropna=False, sort=True):
            beta = frame.loc[:, BETA_COLUMNS].to_numpy(float)
            if np.max(np.abs(beta - beta[0])) > 1.0e-12:
                raise ValueError("symmetry expansion created conflicting labels at one xyz")
        result = result.drop_duplicates(list(XYZ_COLUMNS), keep="first").reset_index(drop=True)
    return result, pd.DataFrame(rejected)


__all__ = [
    "AnnularPolicy",
    "annular_volume_mm3",
    "build_annular_target_registry",
    "build_nested_annular_profiles",
    "deterministic_target_sample",
    "expand_annular_labels",
    "intersect_support",
    "raw_annular_profile",
    "root_connector_voxels",
    "smooth_annular_profile",
]
