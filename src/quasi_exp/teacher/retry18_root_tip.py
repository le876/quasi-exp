"""Zero-tip discovery, profile construction, and coverage helpers for retry18.

Forward proposals in this module are target-only evidence.  Their beta values
must never be reused as labels, solver seeds, warm starts, or branch hints.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import ndtri
from scipy.stats import qmc

from .retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS


@dataclass(frozen=True)
class RootTipPolicy:
    axial_step_mm: float = 2.0
    radial_step_mm: float = 2.0
    sector_count: int = 16
    maximum_u_mm: float = 60.0
    maximum_rho_mm: float = 220.0
    continuity_gap_mm: float = 10.0
    axis_core_radius_mm: float = 5.0
    angular_arc_mm: float = 5.0

    def __post_init__(self) -> None:
        values = (
            self.axial_step_mm,
            self.radial_step_mm,
            self.maximum_u_mm,
            self.maximum_rho_mm,
            self.continuity_gap_mm,
            self.axis_core_radius_mm,
            self.angular_arc_mm,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values):
            raise ValueError("root-tip distances must be finite and positive")
        if int(self.sector_count) < 4:
            raise ValueError("root-tip sector_count must be at least four")


def weighted_sobol_beta_ball(
    *,
    power: int,
    seed: int,
    maximum_weighted_radius_deg: float,
    beta_weights: Sequence[float],
    beta_bounds_rad: np.ndarray,
) -> np.ndarray:
    """Sample a deterministic uniform ball in the weighted beta RMS metric."""

    weights = np.asarray(tuple(beta_weights), dtype=float)
    bounds = np.asarray(beta_bounds_rad, dtype=float)
    if weights.shape != (6,) or bounds.shape != (6, 2):
        raise ValueError("root-tip beta weights/bounds must have shapes (6,) and (6,2)")
    if np.any(weights <= 0) or maximum_weighted_radius_deg <= 0:
        raise ValueError("root-tip beta weights and radius must be positive")
    unit = qmc.Sobol(d=7, scramble=True, seed=int(seed)).random_base2(int(power))
    normal = ndtri(np.clip(unit[:, :6], 1.0e-12, 1.0 - 1.0e-12))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    weighted_radius = float(maximum_weighted_radius_deg) * unit[:, 6:7] ** (1.0 / 6.0)
    weighted_coordinates_deg = normal * weighted_radius
    beta_deg = weighted_coordinates_deg * math.sqrt(float(np.sum(weights**2))) / weights
    beta = np.deg2rad(beta_deg)
    legal = np.logical_and(beta >= bounds[:, 0], beta <= bounds[:, 1]).all(axis=1)
    return beta[legal]


def add_tip_coordinates(xyz_m: np.ndarray, *, zero_x_m: float) -> pd.DataFrame:
    xyz = np.asarray(xyz_m, dtype=float).reshape(-1, 3)
    u_mm = 1000.0 * (float(zero_x_m) - xyz[:, 0])
    y_mm = 1000.0 * np.abs(xyz[:, 1])
    z_mm = 1000.0 * np.abs(xyz[:, 2])
    rho_mm = np.hypot(y_mm, z_mm)
    phi = np.arctan2(z_mm, y_mm)
    return pd.DataFrame(
        {
            "x_m": xyz[:, 0],
            "y_m": xyz[:, 1],
            "z_m": xyz[:, 2],
            "u_mm": u_mm,
            "rho_mm": rho_mm,
            "phi_rad": phi,
        }
    )


def common_zero_tip_support(
    pool_a_xyz_m: np.ndarray,
    pool_b_xyz_m: np.ndarray,
    *,
    zero_x_m: float,
    policy: RootTipPolicy,
) -> pd.DataFrame:
    """Return task voxels occupied by both independent proposal pools."""

    def occupied(points: np.ndarray, pool_id: str) -> pd.DataFrame:
        frame = add_tip_coordinates(points, zero_x_m=zero_x_m)
        frame = frame[
            frame["u_mm"].between(0.0, policy.maximum_u_mm, inclusive="both")
            & frame["rho_mm"].between(0.0, policy.maximum_rho_mm, inclusive="both")
        ].copy()
        frame["u_index"] = np.floor(frame["u_mm"] / policy.axial_step_mm).astype(int)
        frame["rho_index"] = np.floor(frame["rho_mm"] / policy.radial_step_mm).astype(int)
        # Phi is not identifiable on the axis.  Requiring all nominal sectors
        # there turns finite proposal density into a false zero-tip hole.
        required = np.ceil(0.5 * np.pi * frame["rho_mm"] / policy.angular_arc_mm).astype(int)
        required = np.clip(required, 1, int(policy.sector_count))
        required = np.where(frame["rho_mm"] < policy.axis_core_radius_mm, 1, required)
        fraction = np.clip(frame["phi_rad"] / (0.5 * np.pi), 0.0, 1.0 - 1.0e-12)
        frame["required_sector_count"] = required
        frame["sector_index"] = np.floor(fraction * required).astype(int)
        grouped = (
            frame.groupby(["u_index", "rho_index", "required_sector_count", "sector_index"], as_index=False)
            .size()
            .rename(columns={"size": f"{pool_id}_count"})
        )
        return grouped

    a = occupied(pool_a_xyz_m, "pool_a")
    b = occupied(pool_b_xyz_m, "pool_b")
    common = a.merge(
        b,
        on=["u_index", "rho_index", "required_sector_count", "sector_index"],
        how="inner",
    )
    common["u_center_mm"] = (common["u_index"] + 0.5) * policy.axial_step_mm
    common["rho_center_mm"] = (common["rho_index"] + 0.5) * policy.radial_step_mm
    common["proposal_beta_used_as_label_or_hint"] = False
    return common.sort_values(["u_index", "rho_index", "sector_index"], kind="stable").reset_index(drop=True)


def zero_connected_radial_intervals(
    common_support: pd.DataFrame,
    *,
    policy: RootTipPolicy,
) -> pd.DataFrame:
    """Extract one full-sector, radially contiguous interval at every u bin."""

    required = {"u_index", "rho_index", "sector_index"}
    if not required.issubset(common_support.columns):
        raise ValueError(f"common support missing columns: {sorted(required - set(common_support.columns))}")
    complete = (
        common_support.groupby(["u_index", "rho_index"], as_index=False)
        .agg(observed_sector_count=("sector_index", "nunique"), required_sector_count=("required_sector_count", "max"))
        .loc[lambda frame: frame["observed_sector_count"] >= frame["required_sector_count"]]
        [["u_index", "rho_index", "required_sector_count"]]
    )
    rows: list[dict[str, float | int]] = []
    previous: tuple[int, int] | None = None
    for u_index, part in complete.groupby("u_index", sort=True):
        radial = sorted(map(int, part["rho_index"].unique()))
        runs: list[tuple[int, int]] = []
        for value in radial:
            if not runs or value != runs[-1][1] + 1:
                runs.append((value, value))
            else:
                runs[-1] = (runs[-1][0], value)
        if not runs:
            continue
        if previous is None:
            chosen = min(runs, key=lambda run: (run[0], -(run[1] - run[0]), run[1]))
        else:
            def gap(run: tuple[int, int]) -> tuple[float, int, int]:
                left = max(previous[0], run[0])
                right = min(previous[1], run[1])
                separation = 0 if left <= right else min(abs(run[0] - previous[1]), abs(previous[0] - run[1]))
                return (separation * policy.radial_step_mm, -(run[1] - run[0]), run[0])
            chosen = min(runs, key=gap)
            if gap(chosen)[0] > policy.continuity_gap_mm:
                break
        previous = chosen
        rows.append(
            {
                "u_index": int(u_index),
                "u_center_mm": (int(u_index) + 0.5) * policy.axial_step_mm,
                "inner_radius_mm": chosen[0] * policy.radial_step_mm,
                "outer_radius_mm": (chosen[1] + 1) * policy.radial_step_mm,
                "complete_sector_count": int(policy.sector_count),
                "required_sector_count": int(part.loc[part["rho_index"].eq(chosen[0]), "required_sector_count"].iloc[0]),
                "axis_core_radius_mm": float(policy.axis_core_radius_mm),
            }
        )
    return pd.DataFrame(rows)


def fit_sqrt_u_tip_profile(
    intervals: pd.DataFrame,
    *,
    erosion_mm: float,
) -> pd.DataFrame:
    """Build an anchored piecewise-linear profile in sqrt(u)."""

    if erosion_mm < 0:
        raise ValueError("tip profile erosion must be non-negative")
    profile = intervals.loc[:, ["u_center_mm", "inner_radius_mm", "outer_radius_mm"]].copy()
    profile["inner_radius_mm"] += float(erosion_mm)
    profile["outer_radius_mm"] -= float(erosion_mm)
    profile = profile[profile["outer_radius_mm"] > profile["inner_radius_mm"]].copy()
    anchor = pd.DataFrame([{"u_center_mm": 0.0, "inner_radius_mm": 0.0, "outer_radius_mm": 0.0}])
    profile = pd.concat([anchor, profile], ignore_index=True).drop_duplicates("u_center_mm", keep="first")
    profile = profile.sort_values("u_center_mm", kind="stable").reset_index(drop=True)
    profile["sqrt_u_mm"] = np.sqrt(profile["u_center_mm"].to_numpy(float))
    profile["erosion_mm"] = float(erosion_mm)
    profile["profile_id"] = f"tip_erode_{float(erosion_mm):g}mm"
    return profile


def interpolate_tip_profile(profile: pd.DataFrame, u_mm: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(u_mm, dtype=float)
    knots = np.sqrt(profile["u_center_mm"].to_numpy(float))
    coordinate = np.sqrt(np.clip(query, 0.0, None))
    inner = np.interp(coordinate, knots, profile["inner_radius_mm"].to_numpy(float))
    outer = np.interp(coordinate, knots, profile["outer_radius_mm"].to_numpy(float))
    return inner, outer


def profile_overlap_metrics(tip: pd.DataFrame, annulus: pd.DataFrame) -> Mapping[str, float]:
    start = max(float(tip["u_center_mm"].min()), float(annulus["u_center_mm"].min()))
    end = min(float(tip["u_center_mm"].max()), float(annulus["u_center_mm"].max()))
    if end <= start:
        return {"axial_overlap_mm": 0.0, "radial_overlap_minimum_mm": 0.0, "radial_overlap_p50_mm": 0.0}
    query = np.linspace(start, end, max(2, int(math.ceil(end - start)) + 1))
    tip_inner, tip_outer = interpolate_tip_profile(tip, query)
    ann_inner = np.interp(query, annulus["u_center_mm"], annulus["inner_radius_mm"])
    ann_outer = np.interp(query, annulus["u_center_mm"], annulus["outer_radius_mm"])
    thickness = np.maximum(0.0, np.minimum(tip_outer, ann_outer) - np.maximum(tip_inner, ann_inner))
    positive = thickness[thickness > 0]
    if not len(positive):
        return {"axial_overlap_mm": 0.0, "radial_overlap_minimum_mm": 0.0, "radial_overlap_p50_mm": 0.0}
    axial = float(np.count_nonzero(thickness > 0) - 1)
    return {
        "axial_overlap_mm": axial,
        "radial_overlap_minimum_mm": float(np.min(positive)),
        "radial_overlap_p50_mm": float(np.percentile(positive, 50)),
    }


def sample_tip_volume(
    profile: pd.DataFrame,
    *,
    power: int,
    seed: int,
    zero_x_m: float,
    pool_id: str,
) -> pd.DataFrame:
    """Sample a continuous target-only volume using area-correct rho sampling."""

    unit = qmc.Sobol(d=3, scramble=True, seed=int(seed)).random_base2(int(power))
    u_max = float(profile["u_center_mm"].max())
    u_mm = np.maximum(unit[:, 0] * u_max, np.finfo(float).eps)
    inner, outer = interpolate_tip_profile(profile, u_mm)
    rho_mm = np.sqrt(inner**2 + unit[:, 1] * np.maximum(0.0, outer**2 - inner**2))
    phi = 0.5 * np.pi * unit[:, 2]
    frame = pd.DataFrame(
        {
            "target_id": [f"{pool_id}:{index:07d}" for index in range(len(unit))],
            "x_m": float(zero_x_m) - u_mm / 1000.0,
            "y_m": rho_mm * np.cos(phi) / 1000.0,
            "z_m": rho_mm * np.sin(phi) / 1000.0,
            "u_mm": u_mm,
            "rho_mm": rho_mm,
            "phi_rad": phi,
            "pool_id": str(pool_id),
            "proposal_beta_label_eligible": False,
            "proposal_beta_seed_eligible": False,
            "proposal_beta_warm_start_eligible": False,
            "proposal_beta_branch_hint_eligible": False,
        }
    )
    return frame


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


def classify_root_connectors(
    labels: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    residual_maximum_mm: float,
    beta_bounds_rad: np.ndarray,
) -> pd.DataFrame:
    """Classify historical connectors row-by-row for retry18 supervision."""

    merged = targets.merge(labels, on="target_id", how="left", suffixes=("", "_label"), validate="one_to_one")
    beta = merged.reindex(columns=BETA_COLUMNS).to_numpy(float)
    finite = np.isfinite(beta).all(axis=1)
    bounds = np.asarray(beta_bounds_rad, dtype=float)
    bounded = finite & np.logical_and(beta >= bounds[:, 0], beta <= bounds[:, 1]).all(axis=1)
    residual = merged.get("fk_residual_mm", pd.Series(np.inf, index=merged.index)).to_numpy(float)
    eligible = bounded & np.isfinite(residual) & (residual <= float(residual_maximum_mm))
    merged["supervision_eligible"] = eligible
    merged["root_connector_class"] = np.where(eligible, "root_bridge_supervision", "connector_only")
    merged.loc[~finite, "root_connector_class"] = "rejected"
    return merged
