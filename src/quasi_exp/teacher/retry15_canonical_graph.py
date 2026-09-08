"""Retry15 Omega600 domain discovery and canonical-graph helpers.

This module deliberately owns only the new retry15 contracts.  Historical
candidate, symmetry and Student implementations remain immutable.  Proposal
beta values are accepted only by :func:`workspace_probe`; every label-facing
function consumes task-space targets and an independently generated seed bank.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .exploration_qualification import weighted_beta_rms_deg
from .retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS, stable_id, transform_beta, transform_xyz


OMEGA600_LENGTH_MM = 600.0
OMEGA600_X0_M = 1.2154980000000004
OMEGA600_X_MIN_M = 0.6154980000000004


@dataclass(frozen=True)
class Omega600Contract:
    x0_m: float = OMEGA600_X0_M
    x_min_m: float = OMEGA600_X_MIN_M
    length_mm: float = OMEGA600_LENGTH_MM

    def __post_init__(self) -> None:
        values = (self.x0_m, self.x_min_m, self.length_mm)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Omega600 values must be finite")
        observed = 1000.0 * (float(self.x0_m) - float(self.x_min_m))
        if not math.isclose(observed, float(self.length_mm), abs_tol=1.0e-9):
            raise ValueError("Omega600 interval must equal the registered length")
        if not math.isclose(float(self.length_mm), OMEGA600_LENGTH_MM, abs_tol=1.0e-12):
            raise ValueError("retry15 axial interval must remain 600 mm")


@dataclass(frozen=True)
class ProbePolicy:
    axial_step_mm: float = 10.0
    radial_step_mm: float = 10.0
    full_sector_count: int = 16
    radial_margin_mm: float = 10.0
    slope_limit_mm: float = 20.0
    allow_single_voxel_closing: bool = True

    def __post_init__(self) -> None:
        if self.axial_step_mm <= 0 or self.radial_step_mm <= 0:
            raise ValueError("probe steps must be positive")
        if self.full_sector_count < 4 or self.full_sector_count % 4:
            raise ValueError("full_sector_count must be a multiple of four")
        if self.radial_margin_mm < 0 or self.slope_limit_mm <= 0:
            raise ValueError("probe margin and slope must be valid")


def _finite_xyz(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite shape (N, 3)")
    return result


def _probe_keys(
    xyz_m: np.ndarray,
    *,
    contract: Omega600Contract,
    policy: ProbePolicy,
) -> pd.DataFrame:
    xyz = _finite_xyz(xyz_m, name="proposal xyz")
    u_mm = 1000.0 * (float(contract.x0_m) - xyz[:, 0])
    rho_mm = 1000.0 * np.hypot(xyz[:, 1], xyz[:, 2])
    phi = np.mod(np.arctan2(xyz[:, 2], xyz[:, 1]), 2.0 * np.pi)
    axial_count = int(round(contract.length_mm / policy.axial_step_mm))
    inside = (u_mm >= 0.0) & (u_mm <= contract.length_mm + 1.0e-9)
    result = pd.DataFrame(
        {
            "x_m": xyz[inside, 0],
            "y_m": xyz[inside, 1],
            "z_m": xyz[inside, 2],
            "u_mm": u_mm[inside],
            "rho_mm": rho_mm[inside],
            "phi_rad": phi[inside],
        }
    )
    result["u_index"] = np.minimum(
        axial_count - 1,
        np.floor(result["u_mm"].to_numpy(float) / policy.axial_step_mm).astype(int),
    )
    result["rho_index"] = np.floor(
        result["rho_mm"].to_numpy(float) / policy.radial_step_mm
    ).astype(int)
    result["sector_index"] = np.minimum(
        policy.full_sector_count - 1,
        np.floor(
            result["phi_rad"].to_numpy(float)
            / (2.0 * np.pi)
            * policy.full_sector_count
        ).astype(int),
    )
    return result


def _isolated_holes(keys: set[tuple[int, int, int]], *, sectors: int) -> set[tuple[int, int, int]]:
    if not keys:
        return set()
    candidates: set[tuple[int, int, int]] = set()
    max_u = max(key[0] for key in keys)
    max_rho = max(key[1] for key in keys)
    for u in range(1, max_u):
        for rho in range(1, max_rho):
            for sector in range(sectors):
                key = (u, rho, sector)
                if key in keys:
                    continue
                neighbours = {
                    (u - 1, rho, sector),
                    (u + 1, rho, sector),
                    (u, rho - 1, sector),
                    (u, rho + 1, sector),
                    (u, rho, (sector - 1) % sectors),
                    (u, rho, (sector + 1) % sectors),
                }
                if neighbours <= keys:
                    candidates.add(key)
    # Adjacent holes represent a corridor or a patch, not an isolated voxel.
    accepted = set(candidates)
    for u, rho, sector in candidates:
        adjacent = {
            (u - 1, rho, sector),
            (u + 1, rho, sector),
            (u, rho - 1, sector),
            (u, rho + 1, sector),
            (u, rho, (sector - 1) % sectors),
            (u, rho, (sector + 1) % sectors),
        }
        if adjacent & candidates:
            accepted.discard((u, rho, sector))
    return accepted


def workspace_probe(
    proposal_a_xyz_m: np.ndarray,
    proposal_b_xyz_m: np.ndarray,
    *,
    contract: Omega600Contract | None = None,
    policy: ProbePolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the frozen two-pool empirical denominator and raw radius profile."""

    domain = Omega600Contract() if contract is None else contract
    options = ProbePolicy() if policy is None else policy
    a = _probe_keys(proposal_a_xyz_m, contract=domain, policy=options)
    b = _probe_keys(proposal_b_xyz_m, contract=domain, policy=options)
    group = ["u_index", "rho_index", "sector_index"]
    ca = a.groupby(group, sort=True).size().rename("proposal_count_a")
    cb = b.groupby(group, sort=True).size().rename("proposal_count_b")
    support = pd.concat([ca, cb], axis=1).fillna(0).astype(int).reset_index()
    # At rho=0 the azimuth is undefined.  Collapse the axis-core radial cell,
    # then expose the same empirical support to every angular sector so a
    # coordinate singularity cannot manufacture an angular hole.
    axis_a = a.loc[a["rho_index"].eq(0)].groupby("u_index", sort=True).size()
    axis_b = b.loc[b["rho_index"].eq(0)].groupby("u_index", sort=True).size()
    axis_rows = []
    for u_index in sorted(set(axis_a.index) & set(axis_b.index)):
        for sector_index in range(options.full_sector_count):
            axis_rows.append(
                {
                    "u_index": int(u_index),
                    "rho_index": 0,
                    "sector_index": int(sector_index),
                    "proposal_count_a": int(axis_a.loc[u_index]),
                    "proposal_count_b": int(axis_b.loc[u_index]),
                }
            )
    if axis_rows:
        axis = pd.DataFrame(axis_rows)
        support = pd.concat([support, axis], ignore_index=True, sort=False).sort_values(
            [*group, "proposal_count_a", "proposal_count_b"],
            ascending=[True, True, True, False, False],
            kind="stable",
        ).drop_duplicates(group, keep="first")
    support["supported_in_both"] = (
        support["proposal_count_a"].gt(0) & support["proposal_count_b"].gt(0)
    )
    robust_keys = {
        tuple(map(int, row))
        for row in support.loc[support["supported_in_both"], group].itertuples(index=False, name=None)
    }
    holes = (
        _isolated_holes(robust_keys, sectors=options.full_sector_count)
        if options.allow_single_voxel_closing
        else set()
    )
    if holes:
        hole_frame = pd.DataFrame(sorted(holes), columns=group)
        hole_frame["proposal_count_a"] = 0
        hole_frame["proposal_count_b"] = 0
        hole_frame["supported_in_both"] = False
        support = pd.concat([support, hole_frame], ignore_index=True, sort=False)
    support["isolated_hole_closed"] = [
        tuple(map(int, row)) in holes
        for row in support.loc[:, group].itertuples(index=False, name=None)
    ]
    support["robust_supported"] = support["supported_in_both"] | support["isolated_hole_closed"]
    support["u_center_mm"] = (support["u_index"] + 0.5) * options.axial_step_mm
    support["rho_center_mm"] = (support["rho_index"] + 0.5) * options.radial_step_mm
    support["voxel_id"] = [
        stable_id("retry15_outer_voxel", *row)
        for row in support.loc[:, group].itertuples(index=False, name=None)
    ]
    support.sort_values(group, inplace=True, kind="stable")
    support.reset_index(drop=True, inplace=True)

    axial_count = int(round(domain.length_mm / options.axial_step_mm))
    rows: list[dict[str, Any]] = []
    robust = support[support["robust_supported"]]
    by = {
        (int(u), int(sector)): set(map(int, frame["rho_index"]))
        for (u, sector), frame in robust.groupby(["u_index", "sector_index"], sort=True)
    }
    for u in range(axial_count):
        sector_radii: list[float] = []
        for sector in range(options.full_sector_count):
            occupied = by.get((u, sector), set())
            radial = 0
            while radial in occupied:
                radial += 1
            sector_radii.append(radial * options.radial_step_mm)
        raw = max(0.0, min(sector_radii) - options.radial_margin_mm)
        rows.append(
            {
                "u_index": u,
                "u_center_mm": (u + 0.5) * options.axial_step_mm,
                "raw_sector_min_radius_mm": raw,
                "all_sector_supported": bool(raw > 0.0),
            }
        )
    profile = pd.DataFrame(rows)
    values = profile["raw_sector_min_radius_mm"].to_numpy(float)
    values[0] = 0.0
    values = slope_clip(values, limit_mm=options.slope_limit_mm, force_zero=True)
    profile["radius_mm"] = values
    profile["profile_id"] = "retry15_raw_robust_profile"
    return support, profile


def slope_clip(values: Sequence[float], *, limit_mm: float, force_zero: bool = True) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("radius profile must be non-empty, finite and non-negative")
    if not math.isfinite(float(limit_mm)) or limit_mm <= 0:
        raise ValueError("slope limit must be positive")
    if force_zero:
        result[0] = 0.0
    for _ in range(max(2, 2 * len(result))):
        before = result.copy()
        for index in range(1, len(result)):
            result[index] = min(result[index], result[index - 1] + limit_mm)
        for index in range(len(result) - 2, -1, -1):
            result[index] = min(result[index], result[index + 1] + limit_mm)
        if force_zero:
            result[0] = 0.0
        if np.array_equal(before, result):
            break
    return result


def capped_profile(
    raw_profile: pd.DataFrame,
    *,
    ceiling_mm: float,
    shrink_mm: float,
    slope_limit_mm: float,
) -> pd.DataFrame:
    if ceiling_mm < 0 or shrink_mm < 0:
        raise ValueError("profile ceiling and shrink must be non-negative")
    required = {"u_index", "u_center_mm", "radius_mm"}
    if not required <= set(raw_profile):
        raise ValueError("raw profile is missing required columns")
    result = raw_profile.loc[:, ["u_index", "u_center_mm"]].copy()
    values = np.maximum(
        0.0,
        np.minimum(raw_profile["radius_mm"].to_numpy(float), float(ceiling_mm))
        - float(shrink_mm),
    )
    result["radius_mm"] = slope_clip(values, limit_mm=slope_limit_mm, force_zero=True)
    result["ceiling_mm"] = float(ceiling_mm)
    result["shrink_mm"] = float(shrink_mm)
    result["profile_id"] = f"retry15_c{float(ceiling_mm):07.2f}_d{float(shrink_mm):05.2f}"
    return result


def quotient_volume_mm3(profile: pd.DataFrame, *, axial_step_mm: float) -> float:
    radius = profile["radius_mm"].to_numpy(float)
    return float(np.sum(0.25 * np.pi * np.square(radius) * float(axial_step_mm)))


def target_spacing_mm(profile: pd.DataFrame, *, target_budget: int, axial_step_mm: float) -> float:
    if int(target_budget) < 1:
        raise ValueError("target budget must be positive")
    volume = quotient_volume_mm3(profile, axial_step_mm=axial_step_mm)
    return float(np.clip((volume / int(target_budget)) ** (1.0 / 3.0), 8.0, 15.0))


def _stable_xyz_key(xyz: Sequence[float]) -> tuple[int, int, int]:
    return tuple(np.rint(np.asarray(xyz, dtype=float) * 1.0e9).astype(np.int64).tolist())


def _circle_levels(profile: pd.DataFrame) -> tuple[int, int, int] | None:
    eligible = profile.loc[0.9 * profile["radius_mm"].to_numpy(float) >= 100.0, "u_index"].astype(int).tolist()
    for left_pos, left in enumerate(eligible):
        for right in reversed(eligible):
            if right - left < 20:
                continue
            middle_candidates = [value for value in eligible if value - left >= 10 and right - value >= 10]
            if middle_candidates:
                midpoint = 0.5 * (left + right)
                middle = min(middle_candidates, key=lambda value: (abs(value - midpoint), value))
                return int(left), int(middle), int(right)
        if left_pos > 0:
            break
    return None


def build_target_registry(
    profile: pd.DataFrame,
    proposal_xyz_m: np.ndarray,
    *,
    contract: Omega600Contract | None = None,
    axial_step_mm: float = 10.0,
    target_budget: int = 2500,
    maximum_circle_step_mm: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    """Build a quotient registry without deleting required voxels or circles."""

    domain = Omega600Contract() if contract is None else contract
    proposals = _finite_xyz(proposal_xyz_m, name="proposal xyz")
    spacing = target_spacing_mm(profile, target_budget=target_budget, axial_step_mm=axial_step_mm)
    radius_by_u = dict(zip(profile["u_index"].astype(int), profile["radius_mm"].astype(float), strict=True))
    records: dict[tuple[int, int, int], dict[str, Any]] = {}

    def add(xyz: Sequence[float], *, role: str, mandatory: bool, u_index: int, circle_id: str = "") -> str:
        point = np.asarray(xyz, dtype=float).reshape(3)
        key = _stable_xyz_key(point)
        target_id = stable_id("retry15_target", *key)
        row = {
            "target_id": target_id,
            "x_m": float(point[0]),
            "y_m": float(point[1]),
            "z_m": float(point[2]),
            "u_mm": float(1000.0 * (domain.x0_m - point[0])),
            "rho_mm": float(1000.0 * np.hypot(point[1], point[2])),
            "u_index": int(u_index),
            "target_role": str(role),
            "mandatory": bool(mandatory),
            "circle_id": str(circle_id),
            "proposal_beta_label_eligible": False,
            "proposal_beta_seed_eligible": False,
            "proposal_beta_warm_start_eligible": False,
            "proposal_beta_branch_hint_eligible": False,
        }
        previous = records.get(key)
        if previous is None or (mandatory and not bool(previous["mandatory"])):
            records[key] = row
        elif circle_id and not previous["circle_id"]:
            previous.update({"target_role": role, "mandatory": True, "circle_id": circle_id})
        return target_id

    # Exact zero and a complete 10 mm axis core make the 600 mm axial claim explicit.
    add((domain.x0_m, 0.0, 0.0), role="exact_zero", mandatory=True, u_index=0)
    axial_count = int(round(domain.length_mm / axial_step_mm))
    for u_index in range(axial_count):
        u_mm = (u_index + 0.5) * axial_step_mm
        add((domain.x0_m - u_mm / 1000.0, 0.0, 0.0), role="axis_core", mandatory=True, u_index=u_index)

    # Uniform quotient Cartesian cells.  These are denominator representatives;
    # they are never farthest-point downsampled after construction.
    for u_index in range(axial_count):
        radius = float(radius_by_u.get(u_index, 0.0))
        if radius <= 0.0:
            continue
        u_mm = (u_index + 0.5) * axial_step_mm
        coords = np.arange(0.0, radius + 1.0e-9, spacing)
        for y_mm in coords:
            for z_mm in coords:
                if math.hypot(float(y_mm), float(z_mm)) > radius + 1.0e-9:
                    continue
                role = "seam" if y_mm == 0.0 or z_mm == 0.0 else "interior"
                add(
                    (domain.x0_m - u_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0),
                    role=role,
                    mandatory=role == "seam",
                    u_index=u_index,
                )

    circle_levels = _circle_levels(profile)
    edge_rows: list[dict[str, Any]] = []
    circle_rows: list[dict[str, Any]] = []
    if circle_levels is not None:
        for level_rank, u_index in enumerate(circle_levels):
            u_mm = float(profile.loc[profile["u_index"].eq(u_index), "u_center_mm"].iloc[0])
            radius_limit = float(radius_by_u[u_index])
            for radial_rank, fraction in enumerate((0.5, 0.75, 0.9)):
                radius_mm = float(fraction * radius_limit)
                circle_id = f"retry15_circle_u{u_index:03d}_r{radial_rank}"
                full_segments = max(16, int(math.ceil(2.0 * np.pi * radius_mm / maximum_circle_step_mm)))
                full_segments += (-full_segments) % 4
                phases = np.linspace(0.0, 0.5 * np.pi, full_segments // 4 + 1)
                ids: list[str] = []
                for phase_index, phase in enumerate(phases):
                    target_id = add(
                        (
                            domain.x0_m - u_mm / 1000.0,
                            radius_mm * math.cos(float(phase)) / 1000.0,
                            radius_mm * math.sin(float(phase)) / 1000.0,
                        ),
                        role="heldout_circle",
                        mandatory=True,
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
                            "target_id": target_id,
                            "held_out": True,
                        }
                    )
                for left, right in zip(ids[:-1], ids[1:], strict=True):
                    edge_rows.append(
                        {
                            "edge_id": stable_id("retry15_circle_edge", circle_id, left, right),
                            "left_target_id": left,
                            "right_target_id": right,
                            "edge_type": "circle",
                            "circle_id": circle_id,
                            "mandatory": True,
                        }
                    )

    registry = pd.DataFrame(records.values()).sort_values(
        ["mandatory", "u_index", "rho_mm", "target_id"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    tree = cKDTree(proposals)
    distance_m, nearest = tree.query(registry.loc[:, XYZ_COLUMNS].to_numpy(float), k=1)
    registry["proposal_service_mm"] = 1000.0 * np.asarray(distance_m, dtype=float)
    registry["proposal_service_pass"] = registry["proposal_service_mm"].le(spacing + 1.0e-9)
    registry["nearest_proposal_index"] = np.asarray(nearest, dtype=np.int64)
    registry["required_voxel_id"] = registry["target_id"]
    registry["outside_domain"] = False
    registry["target_ordinal"] = np.arange(len(registry), dtype=np.int64)
    circles = pd.DataFrame(circle_rows)
    edges = pd.DataFrame(edge_rows)
    summary = {
        "target_count": int(len(registry)),
        "target_budget": int(target_budget),
        "target_spacing_mm": float(spacing),
        "budget_feasible": bool(len(registry) <= int(target_budget)),
        "mandatory_target_count": int(registry["mandatory"].sum()),
        "circle_level_count": 0 if circle_levels is None else 3,
        "registered_circle_count": int(circles["circle_id"].nunique()) if len(circles) else 0,
        "large_circle_count": int(circles.loc[circles["radius_mm"].ge(100.0), "circle_id"].nunique()) if len(circles) else 0,
        "proposal_service_pass_fraction": float(registry["proposal_service_pass"].mean()),
        "complete_600mm_axis_core": bool(registry["target_role"].eq("axis_core").sum() == axial_count),
    }
    return registry, edges, circles, summary


def select_budget_profile(
    raw_profile: pd.DataFrame,
    proposal_xyz_m: np.ndarray,
    *,
    target_budget: int,
    axial_step_mm: float = 10.0,
    ceiling_step_mm: float = 5.0,
    slope_limit_mm: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    maximum = float(np.max(raw_profile["radius_mm"].to_numpy(float)))
    best: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]] | None = None
    ceiling = 0.0
    while ceiling <= maximum + 1.0e-9:
        profile = capped_profile(
            raw_profile,
            ceiling_mm=ceiling,
            shrink_mm=0.0,
            slope_limit_mm=slope_limit_mm,
        )
        registry, edges, circles, summary = build_target_registry(
            profile,
            proposal_xyz_m,
            axial_step_mm=axial_step_mm,
            target_budget=target_budget,
        )
        eligible = bool(
            summary["budget_feasible"]
            and summary["complete_600mm_axis_core"]
            and summary["registered_circle_count"] == 9
            and summary["large_circle_count"] >= 3
        )
        if eligible:
            best = profile, registry, edges, circles, dict(summary)
        ceiling += ceiling_step_mm
    if best is None:
        raise ValueError("no budget-feasible profile satisfies the registered 600 mm / 100 mm circle objective")
    profile, registry, edges, circles, summary = best
    summary = dict(summary)
    summary["selected_ceiling_mm"] = float(profile["ceiling_mm"].iloc[0])
    summary["selection_rule"] = "largest_5mm_ceiling_with_complete_registry_within_budget"
    return profile, registry, edges, circles, summary


def mutual_knn_edges(
    targets: pd.DataFrame,
    explicit_edges: pd.DataFrame,
    *,
    k: int,
    maximum_distance_mm: float,
) -> pd.DataFrame:
    if int(k) < 1 or maximum_distance_mm <= 0:
        raise ValueError("graph k and maximum distance must be positive")
    xyz = targets.loc[:, XYZ_COLUMNS].to_numpy(float)
    ids = targets["target_id"].astype(str).tolist()
    count = len(targets)
    tree = cKDTree(xyz)
    _distance, indices = tree.query(xyz, k=min(count, int(k) + 1))
    neighbour_sets = [set(map(int, np.atleast_1d(row)[1:])) for row in np.atleast_2d(indices)]
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for left in range(count):
        for right in neighbour_sets[left]:
            if left >= right or left not in neighbour_sets[right]:
                continue
            distance_mm = float(np.linalg.norm(xyz[left] - xyz[right]) * 1000.0)
            if distance_mm > maximum_distance_mm + 1.0e-9:
                continue
            pair = tuple(sorted((ids[left], ids[right])))
            rows[pair] = {
                "edge_id": stable_id("retry15_knn_edge", *pair),
                "left_target_id": pair[0],
                "right_target_id": pair[1],
                "edge_type": "mutual_knn",
                "distance_mm": distance_mm,
                "mandatory": False,
            }
    for row in explicit_edges.to_dict("records"):
        pair = tuple(sorted((str(row["left_target_id"]), str(row["right_target_id"]))))
        left = ids.index(pair[0])
        right = ids.index(pair[1])
        values = dict(row)
        values["left_target_id"], values["right_target_id"] = pair
        values["distance_mm"] = float(np.linalg.norm(xyz[left] - xyz[right]) * 1000.0)
        rows[pair] = values
    return pd.DataFrame(rows.values()).sort_values(
        ["edge_type", "left_target_id", "right_target_id"], kind="stable"
    ).reset_index(drop=True)


def legal_candidate_clusters(
    candidates: pd.DataFrame,
    *,
    residual_maximum_mm: float = 3.0,
    cluster_threshold_deg: float = 0.5,
    weights: Sequence[float] = (4, 4, 2, 2, 1, 1),
) -> pd.DataFrame:
    required = {"target_id", "candidate_id", *BETA_COLUMNS, "solver_success", "bounds_pass", "fk_residual_mm"}
    if not required <= set(candidates):
        raise ValueError("candidate frame is missing legality columns")
    legal = candidates[
        candidates["solver_success"].astype(bool)
        & candidates["bounds_pass"].astype(bool)
        & candidates["fk_residual_mm"].le(float(residual_maximum_mm))
    ].copy()
    if legal.empty:
        legal["cluster_id"] = pd.Series(dtype=str)
        return legal
    output: list[pd.DataFrame] = []
    for target_id, frame in legal.groupby("target_id", sort=True):
        ordered = frame.sort_values(
            ["fk_residual_mm", "min_margin_deg", "candidate_id"],
            ascending=[True, False, True], kind="stable"
        ).copy()
        representatives: list[np.ndarray] = []
        cluster_ids: list[str] = []
        for beta in ordered.loc[:, BETA_COLUMNS].to_numpy(float):
            cluster = None
            for index, representative in enumerate(representatives):
                gap = float(weighted_beta_rms_deg(beta, representative, weights))
                if gap <= cluster_threshold_deg:
                    cluster = index
                    break
            if cluster is None:
                representatives.append(beta.copy())
                cluster = len(representatives) - 1
            cluster_ids.append(f"{target_id}:cluster:{cluster:02d}")
        ordered["cluster_id"] = cluster_ids
        output.append(ordered.drop_duplicates("cluster_id", keep="first"))
    return pd.concat(output, ignore_index=True, sort=False)


def select_t1(
    candidates: pd.DataFrame,
    *,
    weights: Sequence[float] = (4, 4, 2, 2, 1, 1),
) -> pd.DataFrame:
    legal = legal_candidate_clusters(candidates, weights=weights)
    if legal.empty:
        return legal
    beta = legal.loc[:, BETA_COLUMNS].to_numpy(float)
    legal["unary_weighted_deg"] = np.asarray(
        weighted_beta_rms_deg(beta, np.zeros_like(beta), weights), dtype=float
    )
    return legal.sort_values(
        ["target_id", "unary_weighted_deg", "min_margin_deg", "candidate_id"],
        ascending=[True, True, False, True], kind="stable"
    ).drop_duplicates("target_id", keep="first").reset_index(drop=True)


def _pair_cost(left: np.ndarray, right: np.ndarray, *, weights: Sequence[float], tau_deg: float) -> float:
    gap = float(weighted_beta_rms_deg(left, right, weights))
    return min(gap * gap, float(tau_deg) ** 2)


def select_t2(
    candidates: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    pairwise_lambda: float,
    tau_deg: float = 7.0,
    weights: Sequence[float] = (4, 4, 2, 2, 1, 1),
    maximum_sweeps: int = 30,
) -> pd.DataFrame:
    """Deterministic robust graph selection; T2 never creates candidates."""

    if pairwise_lambda < 0 or tau_deg <= 0 or maximum_sweeps < 1:
        raise ValueError("invalid T2 policy")
    legal = legal_candidate_clusters(candidates, weights=weights)
    if legal.empty or pairwise_lambda == 0:
        return select_t1(candidates, weights=weights).assign(
            teacher="T2", pairwise_lambda=float(pairwise_lambda)
        )
    grouped = {
        str(target): frame.sort_values("candidate_id", kind="stable").reset_index(drop=True)
        for target, frame in legal.groupby("target_id", sort=True)
    }
    node_ids = sorted(grouped)
    adjacency: dict[str, set[str]] = {target: set() for target in node_ids}
    edge_weights: dict[tuple[str, str], float] = {}
    for row in edges.to_dict("records"):
        left, right = str(row["left_target_id"]), str(row["right_target_id"])
        if left in grouped and right in grouped and left != right:
            pair = tuple(sorted((left, right)))
            distance = float(row.get("distance_mm", 1.0))
            if not math.isfinite(distance) or distance < 0:
                raise ValueError("T2 edge distances must be finite and non-negative")
            edge_weights[pair] = min(distance, edge_weights.get(pair, math.inf))
    valid_edges = sorted(edge_weights)
    for left, right in valid_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unary = {
        target: np.square(
            np.asarray(
                weighted_beta_rms_deg(
                    frame.loc[:, BETA_COLUMNS].to_numpy(float),
                    np.zeros((len(frame), 6)),
                    weights,
                ),
                dtype=float,
            )
        )
        for target, frame in grouped.items()
    }
    beta = {target: frame.loc[:, BETA_COLUMNS].to_numpy(float) for target, frame in grouped.items()}
    pair_cost: dict[tuple[str, str], np.ndarray] = {}
    for left, right in valid_edges:
        matrix = np.empty((len(grouped[left]), len(grouped[right])), dtype=float)
        for i, left_beta in enumerate(beta[left]):
            for j, right_beta in enumerate(beta[right]):
                matrix[i, j] = _pair_cost(left_beta, right_beta, weights=weights, tau_deg=tau_deg)
        pair_cost[(left, right)] = matrix

    # Build a deterministic minimum spanning forest with Kruskal, then solve
    # the exact unary + pairwise objective on every tree by dynamic
    # programming.  This is the registered T2 initialization; full-graph ICM
    # below may only improve or preserve that objective.
    parent = {target: target for target in node_ids}

    def find(target: str) -> str:
        while parent[target] != target:
            parent[target] = parent[parent[target]]
            target = parent[target]
        return target

    tree_adjacency: dict[str, set[str]] = {target: set() for target in node_ids}
    for left, right in sorted(
        valid_edges,
        key=lambda pair: (edge_weights[pair], pair[0], pair[1]),
    ):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        tree_adjacency[left].add(right)
        tree_adjacency[right].add(left)

    initial: dict[str, int] = {}
    mst_order: list[str] = []
    visited: set[str] = set()
    for root in node_ids:
        if root in visited:
            continue
        traversal: list[str] = []
        tree_parent: dict[str, str | None] = {root: None}
        stack = [root]
        while stack:
            target = stack.pop()
            if target in visited:
                continue
            visited.add(target)
            traversal.append(target)
            children = sorted(tree_adjacency[target] - {tree_parent[target]}, reverse=True)
            for child in children:
                tree_parent[child] = target
                stack.append(child)
        mst_order.extend(traversal)
        subtree_cost: dict[str, np.ndarray] = {}
        backtrack: dict[tuple[str, str], np.ndarray] = {}
        for target in reversed(traversal):
            costs = unary[target].copy()
            for child in sorted(tree_adjacency[target] - {tree_parent[target]}):
                pair = tuple(sorted((target, child)))
                matrix = pair_cost[pair]
                oriented = matrix if target == pair[0] else matrix.T
                totals = subtree_cost[child][None, :] + float(pairwise_lambda) * oriented
                choices = np.argmin(totals, axis=1).astype(int)
                costs += totals[np.arange(len(costs)), choices]
                backtrack[(target, child)] = choices
            subtree_cost[target] = costs
        initial[root] = int(np.argmin(subtree_cost[root]))
        for target in traversal:
            for child in sorted(tree_adjacency[target] - {tree_parent[target]}):
                initial[child] = int(backtrack[(target, child)][initial[target]])

    digest_order = sorted(
        node_ids,
        key=lambda value: (hashlib.sha256(f"retry15-icm:{value}".encode()).hexdigest(), value),
    )
    orders = (mst_order, list(reversed(mst_order)), digest_order)

    def objective(selection: Mapping[str, int]) -> float:
        value = sum(float(unary[target][selection[target]]) for target in node_ids)
        for left, right in valid_edges:
            value += float(pairwise_lambda) * float(
                pair_cost[(left, right)][selection[left], selection[right]]
            )
        return value

    solutions: list[tuple[float, tuple[str, ...], dict[str, int]]] = []
    for order in orders:
        selection = dict(initial)
        for _ in range(maximum_sweeps):
            changed = False
            for target in order:
                costs = unary[target].copy()
                for neighbour in adjacency[target]:
                    pair = tuple(sorted((target, neighbour)))
                    matrix = pair_cost[pair]
                    if target == pair[0]:
                        costs += float(pairwise_lambda) * matrix[:, selection[neighbour]]
                    else:
                        costs += float(pairwise_lambda) * matrix[selection[neighbour], :]
                chosen = int(np.argmin(costs))
                if chosen != selection[target]:
                    selection[target] = chosen
                    changed = True
            if not changed:
                break
        signature = tuple(grouped[target].iloc[selection[target]]["candidate_id"] for target in node_ids)
        solutions.append((objective(selection), signature, selection))
    _value, _signature, selected = min(solutions, key=lambda item: (item[0], item[1]))
    rows = [grouped[target].iloc[[selected[target]]].copy() for target in node_ids]
    result = pd.concat(rows, ignore_index=True, sort=False)
    result["teacher"] = "T2"
    result["pairwise_lambda"] = float(pairwise_lambda)
    return result


def candidate_attempt_upper_bound(*, target_count: int, seed_budget: int, difficult_target_count: int) -> int:
    if min(target_count, seed_budget, difficult_target_count) < 0 or difficult_target_count > target_count:
        raise ValueError("candidate attempt counts must be non-negative and consistent")
    return int(2 * target_count * seed_budget + 4 * difficult_target_count)


def objective_feasibility_documents(
    *,
    experiment_id: str,
    scientific_source_sha: str,
    config_sha256: str,
    target_registry: Mapping[str, Any],
    circle_registry: Mapping[str, Any],
    target_budget: int,
    solver_attempt_budget: int,
    candidate_seed_budget: int,
    difficult_target_count: int,
    source_artifacts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    target_count = int(target_registry["row_count"])
    circle_count = int(circle_registry["row_count"])
    attempts = candidate_attempt_upper_bound(
        target_count=target_count,
        seed_budget=candidate_seed_budget,
        difficult_target_count=difficult_target_count,
    )
    resources_feasible = target_count <= int(target_budget) and attempts <= int(solver_attempt_budget)
    contract = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "scope": "coverage_trajectory",
        "scientific_source_sha": scientific_source_sha,
        "config_sha256": config_sha256,
        "diagnostic_pilot_allowed": True,
        "primary_objectives": [
            {
                "id": "omega600_domain_coverage",
                "kind": "coverage",
                "metric": "frozen_inner_voxel_coverage",
                "required_for_claim": True,
                "denominator_id": "retry15_required_targets",
                "target": {"operator": ">=", "value": 0.8, "unit": "fraction"},
            },
            {
                "id": "heldout_large_circles",
                "kind": "complete_trajectory",
                "metric": "complete_circle_count",
                "required_for_claim": True,
                "denominator_id": "retry15_registered_circles",
                "target": {"operator": ">=", "value": 9, "unit": "count"},
            },
            {
                "id": "teacher_dataset_size",
                "kind": "dataset_size",
                "metric": "quotient_target_count",
                "required_for_claim": True,
                "denominator_id": "retry15_required_targets",
                "target": {"operator": ">=", "value": target_count, "unit": "count"},
            },
        ],
    }
    denominator = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "frozen_before_launch": True,
        "denominators": [
            {"id": "retry15_required_targets", "kind": "coverage", "unit": "target", "required_count": target_count, "registry": dict(target_registry)},
            {"id": "retry15_registered_circles", "kind": "complete_trajectory", "unit": "circle", "required_count": circle_count, "registry": dict(circle_registry)},
        ],
        "resource_denominators": {
            "required_supervision_vertex_count": target_count,
            "required_logical_edge_count": 0,
            "required_second_parent_certification_count": 0,
        },
    }
    reusable = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "source_artifacts": [dict(value) for value in source_artifacts],
        "eligible_counts": {"supervision_vertices": 0, "connector_only_vertices": 0, "served_coverage_units": 0, "complete_trajectories": 0, "verified_edges": 0, "second_parent_certifications": 0},
        "ineligible_counts": {"proposal_only": target_count, "branch_conflicts": 0, "unused": 0},
        "credit_registry": dict(target_registry),
        "proposal_beta_used_as_label_or_hint": False,
    }
    basis = dict(target_registry)
    lower = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "objective_lower_bounds": [
            {"objective_id": "omega600_domain_coverage", "method": "count_credit", "required_units": target_count, "target_units": int(math.ceil(0.8 * target_count)), "reusable_eligible_units": 0, "maximum_credit_per_new_supervision_vertex": 1, "minimum_resources": {"new_supervision_vertices": int(math.ceil(0.8 * target_count))}, "basis_registry": basis},
            {"objective_id": "heldout_large_circles", "method": "registered_atomic_set_union", "minimum_resources": {"new_supervision_vertices": target_count, "solver_attempts": attempts}, "basis_registry": dict(circle_registry)},
            {"objective_id": "teacher_dataset_size", "method": "count_credit", "required_units": target_count, "target_units": target_count, "reusable_eligible_units": 0, "maximum_credit_per_new_supervision_vertex": 1, "minimum_resources": {"new_supervision_vertices": target_count}, "basis_registry": basis},
        ],
        "resources": {
            "new_supervision_vertices": {"optimistic_minimum": target_count, "registered_budget": int(target_budget), "basis_registry": basis},
            "solver_attempts": {"optimistic_minimum": attempts, "registered_budget": int(solver_attempt_budget), "basis_registry": basis},
        },
        "all_required_objectives_bounded": True,
        "all_required_resources_feasible": bool(resources_feasible),
    }
    schedule = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "row_count_is_stop_condition": False,
        "scheduled_objective_ids": ["omega600_domain_coverage", "heldout_large_circles", "teacher_dataset_size"],
        "entries": [
            {"id": "omega600_inner_registry", "objective_id": "omega600_domain_coverage", "kind": "coverage", "priority": 0, "required_for_claim": True, "requirement_registry": basis, "reserved_resources": {"new_supervision_vertices": target_count, "solver_attempts": attempts}},
            {"id": "omega600_nine_heldout_circles", "objective_id": "heldout_large_circles", "kind": "trajectory", "priority": 0, "required_for_claim": True, "minimum_diameter_mm": 200.0, "full_cycle_required": True, "held_out": True, "requirement_registry": dict(circle_registry), "reserved_resources": {"new_supervision_vertices": 0, "solver_attempts": 0}},
            {"id": "omega600_teacher_dataset", "objective_id": "teacher_dataset_size", "kind": "dataset_size", "priority": 1, "required_for_claim": True, "requirement_registry": basis, "reserved_resources": {"new_supervision_vertices": 0, "solver_attempts": 0}},
        ],
    }
    gate = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "inputs_valid": True,
        "required_objectives_budget_feasible": bool(resources_feasible),
        "status": "feasible" if resources_feasible else "diagnostic_only",
        "claim_bearing_run_authorized": bool(resources_feasible),
        "diagnostic_pilot_authorized": not resources_feasible,
        "failed_objective_ids": [] if resources_feasible else ["omega600_domain_coverage", "heldout_large_circles", "teacher_dataset_size"],
        "reason_codes": [] if resources_feasible else ["REGISTERED_RESOURCE_BUDGET_BELOW_OPTIMISTIC_MINIMUM"],
    }
    return {
        "objective_contract.json": contract,
        "denominator_size.json": denominator,
        "reusable_evidence.json": reusable,
        "budget_lower_bound.json": lower,
        "atomic_objective_schedule.json": schedule,
        "gate.json": gate,
    }


__all__ = [
    "OMEGA600_LENGTH_MM",
    "OMEGA600_X0_M",
    "OMEGA600_X_MIN_M",
    "Omega600Contract",
    "ProbePolicy",
    "build_target_registry",
    "candidate_attempt_upper_bound",
    "capped_profile",
    "legal_candidate_clusters",
    "mutual_knn_edges",
    "objective_feasibility_documents",
    "quotient_volume_mm3",
    "select_budget_profile",
    "select_t1",
    "select_t2",
    "slope_clip",
    "target_spacing_mm",
    "workspace_probe",
]
