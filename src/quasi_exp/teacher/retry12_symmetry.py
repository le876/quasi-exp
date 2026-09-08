"""Retry12 zero-centred symmetry quotient helpers.

The functions in this module are deliberately data-frame oriented so the
scientific identities remain explicit in artifacts and contract tests.
Proposal beta values are never accepted by any label-building function.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc

from quasi_exp.model.sampling import beta_to_theta


XYZ_COLUMNS = ("x_m", "y_m", "z_m")
BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
THETA_COLUMNS = tuple(f"theta{index}_rad" for index in range(1, 31))

SYMMETRY_BETA_SIGNS: dict[str, np.ndarray] = {
    "identity": np.asarray([1, 1, 1, 1, 1, 1], dtype=float),
    "mirror_y": np.asarray([-1, 1, -1, 1, -1, 1], dtype=float),
    "mirror_z": np.asarray([1, -1, 1, -1, 1, -1], dtype=float),
    "rotate_x_180": np.asarray([-1, -1, -1, -1, -1, -1], dtype=float),
}
SYMMETRY_XYZ_SIGNS: dict[str, np.ndarray] = {
    "identity": np.asarray([1, 1, 1], dtype=float),
    "mirror_y": np.asarray([1, -1, 1], dtype=float),
    "mirror_z": np.asarray([1, 1, -1], dtype=float),
    "rotate_x_180": np.asarray([1, -1, -1], dtype=float),
}


def stable_id(prefix: str, *parts: object) -> str:
    payload = ":".join([prefix, *(str(part) for part in parts)])
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def transform_beta(beta: np.ndarray, element: str) -> np.ndarray:
    return np.asarray(beta, dtype=float) * SYMMETRY_BETA_SIGNS[element]


def transform_xyz(xyz: np.ndarray, element: str) -> np.ndarray:
    return np.asarray(xyz, dtype=float) * SYMMETRY_XYZ_SIGNS[element]


def validate_symmetry_group(
    environment: Any,
    beta_samples: np.ndarray,
    *,
    p99_max_mm: float,
    individual_max_mm: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    beta = np.asarray(beta_samples, dtype=float).reshape(-1, 6)
    bounds = np.asarray(environment.bounds, dtype=float)
    base_xyz = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
    records: list[dict[str, Any]] = []
    registry: dict[str, Any] = {}
    for element in SYMMETRY_BETA_SIGNS:
        transformed = transform_beta(beta, element)
        observed = np.asarray(environment.fk(transformed), dtype=float).reshape(-1, 3)
        expected = transform_xyz(base_xyz, element)
        error_mm = np.linalg.norm(observed - expected, axis=1) * 1000.0
        in_bounds = np.all(
            (transformed >= bounds[:, 0] - 1.0e-12)
            & (transformed <= bounds[:, 1] + 1.0e-12),
            axis=1,
        )
        involution = np.max(np.abs(transform_beta(transformed, element) - beta), axis=1)
        for index in range(len(beta)):
            records.append(
                {
                    "sample_index": index,
                    "symmetry_element": element,
                    "equivariance_error_mm": float(error_mm[index]),
                    "transformed_in_bounds": bool(in_bounds[index]),
                    "involution_error_rad": float(involution[index]),
                }
            )
        p99 = float(np.percentile(error_mm, 99)) if len(error_mm) else math.inf
        maximum = float(np.max(error_mm)) if len(error_mm) else math.inf
        passed = bool(
            p99 <= float(p99_max_mm) + 1.0e-12
            and maximum <= float(individual_max_mm) + 1.0e-12
            and in_bounds.all()
            and np.max(involution, initial=0.0) <= 1.0e-12
        )
        registry[element] = {
            "authorized": passed,
            "beta_signs": SYMMETRY_BETA_SIGNS[element].astype(int).tolist(),
            "xyz_signs": SYMMETRY_XYZ_SIGNS[element].astype(int).tolist(),
            "p99_error_mm": p99,
            "max_error_mm": maximum,
        }
    registry["group_composition_verified"] = bool(
        np.array_equal(
            SYMMETRY_BETA_SIGNS["mirror_y"] * SYMMETRY_BETA_SIGNS["mirror_z"],
            SYMMETRY_BETA_SIGNS["rotate_x_180"],
        )
    )
    registry["exact_zero_fixed"] = bool(
        all(np.array_equal(transform_beta(np.zeros(6), key), np.zeros(6)) for key in SYMMETRY_BETA_SIGNS)
    )
    return pd.DataFrame.from_records(records), registry


def sobol_beta_samples(bounds: np.ndarray, *, power: int, seed: int) -> np.ndarray:
    limits = np.asarray(bounds, dtype=float).reshape(6, 2)
    unit = qmc.Sobol(d=6, scramble=True, seed=int(seed)).random_base2(int(power))
    return qmc.scale(unit, limits[:, 0], limits[:, 1])


def reduced_seam_beta_samples(
    bounds: np.ndarray,
    *,
    seam: str,
    power: int,
    seed: int,
) -> np.ndarray:
    limits = np.asarray(bounds, dtype=float).reshape(6, 2)
    if seam == "y_seam":
        free = np.asarray([1, 3, 5], dtype=int)
    elif seam == "z_seam":
        free = np.asarray([0, 2, 4], dtype=int)
    else:
        raise ValueError(f"unsupported seam {seam!r}")
    unit = qmc.Sobol(d=3, scramble=True, seed=int(seed)).random_base2(int(power))
    sampled = qmc.scale(unit, limits[free, 0], limits[free, 1])
    beta = np.zeros((len(sampled), 6), dtype=float)
    beta[:, free] = sampled
    return beta


def proposal_registry(
    xyz: np.ndarray,
    *,
    source: str,
    zero_xyz_m: Sequence[float],
    shell_width_mm: float,
    seam_half_width_mm: float,
    angle_bins: int,
    cell_size_mm: float,
    x_bin_mm: float,
) -> pd.DataFrame:
    points = np.asarray(xyz, dtype=float).reshape(-1, 3).copy()
    points[:, 1:] = np.abs(points[:, 1:])
    zero = np.asarray(zero_xyz_m, dtype=float).reshape(3)
    delta = points - zero
    radius_mm = np.linalg.norm(delta, axis=1) * 1000.0
    rho = np.linalg.norm(points[:, 1:], axis=1)
    phi = np.arctan2(points[:, 2], points[:, 1])
    shell = np.floor(radius_mm / float(shell_width_mm)).astype(np.int64)
    angle = np.minimum(
        np.floor(phi / (0.5 * np.pi) * int(angle_bins)).astype(np.int64),
        int(angle_bins) - 1,
    )
    x_bin = np.floor((points[:, 0] - zero[0]) * 1000.0 / float(x_bin_mm)).astype(np.int64)
    cell = np.floor(points * 1000.0 / float(cell_size_mm)).astype(np.int64)
    cell_id = [f"qcell:{a}:{b}:{c}" for a, b, c in cell]
    seam_m = float(seam_half_width_mm) / 1000.0
    core = radius_mm <= 2.0 * float(shell_width_mm) + 1.0e-12
    y_seam = points[:, 1] <= seam_m + 1.0e-12
    z_seam = points[:, 2] <= seam_m + 1.0e-12
    role = np.where(core, "core", np.where(y_seam, "y_seam_guard", np.where(z_seam, "z_seam_guard", "interior")))
    frame = pd.DataFrame(
        {
            "proposal_id": [stable_id("retry12_target", source, index, value) for index, value in enumerate(cell_id)],
            "proposal_source": source,
            "x_m": points[:, 0],
            "y_m": points[:, 1],
            "z_m": points[:, 2],
            "zero_radius_mm": radius_mm,
            "rho_m": rho,
            "phi_rad": phi,
            "radial_shell_id": shell,
            "angle_bin_id": angle,
            "x_bin_id": x_bin,
            "quotient_parent_id": cell_id,
            "region_role": role,
            "proposal_beta_label_eligible": False,
            "proposal_beta_seed_eligible": False,
            "proposal_beta_warm_start_eligible": False,
            "proposal_beta_tie_break_eligible": False,
            "proposal_beta_branch_hint_eligible": False,
        }
    )
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(XYZ_COLUMNS))


def deduplicate_proposals(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    priority = {"registered_task": 0, "outer_xyz": 1, "sobol_target_only": 2}
    combined["_source_priority"] = combined["proposal_source"].map(priority).fillna(9)
    combined = combined.sort_values(
        ["_source_priority", "zero_radius_mm", "proposal_id"], kind="stable"
    )
    return combined.drop_duplicates("quotient_parent_id", keep="first").drop(columns="_source_priority").reset_index(drop=True)


def spatially_balanced_frontier(
    candidates: pd.DataFrame,
    retained: pd.DataFrame,
    *,
    maximum_step_mm: float,
    count: int,
) -> pd.DataFrame:
    if candidates.empty or retained.empty or count <= 0:
        return candidates.head(0).copy()
    remaining = candidates.copy()
    retained_xyz = retained.loc[:, XYZ_COLUMNS].to_numpy(float)
    tree = cKDTree(retained_xyz)
    distance_m, nearest = tree.query(remaining.loc[:, XYZ_COLUMNS].to_numpy(float), k=1)
    remaining["nearest_source_position"] = nearest.astype(int)
    remaining["nearest_source_distance_mm"] = distance_m * 1000.0
    eligible = remaining[
        remaining["nearest_source_distance_mm"].le(float(maximum_step_mm) + 1.0e-9)
    ].copy()
    if eligible.empty:
        return eligible
    shell_count = retained.get("radial_shell_id", pd.Series(dtype=int)).value_counts().to_dict()
    angle_count = retained.get("angle_bin_id", pd.Series(dtype=int)).value_counts().to_dict()
    x_count = retained.get("x_bin_id", pd.Series(dtype=int)).value_counts().to_dict()
    eligible["_shell_count"] = eligible["radial_shell_id"].map(shell_count).fillna(0)
    eligible["_angle_count"] = eligible["angle_bin_id"].map(angle_count).fillna(0)
    eligible["_x_count"] = eligible["x_bin_id"].map(x_count).fillna(0)
    eligible = eligible.sort_values(
        ["_shell_count", "_angle_count", "_x_count", "nearest_source_distance_mm", "proposal_id"],
        ascending=[True, True, True, False, True],
        kind="stable",
    )
    selected: list[int] = []
    xyz = eligible.loc[:, XYZ_COLUMNS].to_numpy(float)
    min_distance = np.full(len(eligible), np.inf)
    for _ in range(min(int(count), len(eligible))):
        if not selected:
            position = 0
        else:
            last = xyz[selected[-1]]
            min_distance = np.minimum(min_distance, np.linalg.norm(xyz - last, axis=1))
            min_distance[selected] = -1.0
            position = int(np.argmax(min_distance))
        selected.append(position)
    return eligible.iloc[selected].drop(columns=["_shell_count", "_angle_count", "_x_count"]).reset_index(drop=True)


def orbit_class(xyz: Sequence[float], *, exact_tolerance_m: float, zero_xyz_m: Sequence[float]) -> str:
    point = np.asarray(xyz, dtype=float).reshape(3)
    zero = np.asarray(zero_xyz_m, dtype=float).reshape(3)
    if np.linalg.norm(point - zero) <= float(exact_tolerance_m):
        return "exact_zero"
    on_y = abs(point[1]) <= float(exact_tolerance_m)
    on_z = abs(point[2]) <= float(exact_tolerance_m)
    if on_y and on_z:
        return "double_seam_nonzero_abstain"
    if on_y:
        return "y_seam"
    if on_z:
        return "z_seam"
    return "interior"


def expand_symmetry_orbits(
    representatives: pd.DataFrame,
    environment: Any,
    *,
    zero_xyz_m: Sequence[float],
    exact_tolerance_m: float,
    fk_max_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for representative in representatives.to_dict("records"):
        xyz = np.asarray([representative[name] for name in XYZ_COLUMNS], dtype=float)
        beta = np.asarray([representative[name] for name in BETA_COLUMNS], dtype=float)
        kind = orbit_class(xyz, exact_tolerance_m=exact_tolerance_m, zero_xyz_m=zero_xyz_m)
        if kind == "double_seam_nonzero_abstain":
            rejected.append({**representative, "orbit_rejection_reason": kind})
            continue
        elements = {
            "exact_zero": ("identity",),
            "y_seam": ("identity", "mirror_z"),
            "z_seam": ("identity", "mirror_y"),
            "interior": tuple(SYMMETRY_BETA_SIGNS),
        }[kind]
        orbit_id = stable_id("retry12_orbit", representative["fundamental_representative_id"])
        members: list[dict[str, Any]] = []
        valid = True
        for element in elements:
            member_xyz = transform_xyz(xyz, element)
            member_beta = transform_beta(beta, element)
            observed = np.asarray(environment.fk(member_beta), dtype=float).reshape(-1, 3)[0]
            residual_mm = float(np.linalg.norm(observed - member_xyz) * 1000.0)
            bounds = np.asarray(environment.bounds, dtype=float)
            valid = valid and bool(
                np.isfinite(member_beta).all()
                and np.all(member_beta >= bounds[:, 0] - 1.0e-12)
                and np.all(member_beta <= bounds[:, 1] + 1.0e-12)
                and residual_mm <= float(fk_max_mm) + 1.0e-12
            )
            theta = beta_to_theta(member_beta)
            members.append(
                {
                    **representative,
                    **dict(zip(XYZ_COLUMNS, member_xyz, strict=True)),
                    **dict(zip(BETA_COLUMNS, member_beta, strict=True)),
                    **dict(zip(THETA_COLUMNS, theta, strict=True)),
                    "symmetry_orbit_id": orbit_id,
                    "symmetry_element": element,
                    "orbit_size": len(elements),
                    "seam_class": kind,
                    "fk_residual_mm": residual_mm,
                }
            )
        if valid:
            expanded.extend(members)
        else:
            rejected.append({**representative, "orbit_rejection_reason": "invalid_orbit_member"})
    return pd.DataFrame.from_records(expanded), pd.DataFrame.from_records(rejected)


def quotient_macroblock_id(frame: pd.DataFrame, *, block_size_mm: int) -> pd.Series:
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(float).copy()
    xyz[:, 1:] = np.abs(xyz[:, 1:])
    bins = np.floor(xyz * 1000.0 / int(block_size_mm)).astype(np.int64)
    return pd.Series([f"qmacro:{a}:{b}:{c}" for a, b, c in bins], index=frame.index)


def assign_quotient_macroblock_splits(
    frame: pd.DataFrame,
    *,
    block_size_mm: int,
    seed: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["quotient_macroblock_id"] = quotient_macroblock_id(result, block_size_mm=block_size_mm)
    role_by_block: dict[str, str] = {}
    for block in sorted(result["quotient_macroblock_id"].astype(str).unique()):
        value = int(hashlib.sha256(f"{seed}:{block}".encode()).hexdigest(), 16) / float(2**256)
        role_by_block[block] = "train" if value < 0.70 else "validation" if value < 0.85 else "test"
    result["split_role"] = result["quotient_macroblock_id"].map(role_by_block)
    return result


def select_exact_orbit_budget(
    representatives: pd.DataFrame,
    *,
    target_expanded_rows: int,
) -> pd.DataFrame:
    frame = representatives.copy()
    zero = frame[frame["orbit_size"].eq(1)].sort_values("fundamental_representative_id", kind="stable").head(1)
    seams = frame[frame["orbit_size"].eq(2)].sort_values(
        ["mandatory_seam_representative", "selection_priority", "fundamental_representative_id"],
        ascending=[False, True, True], kind="stable"
    )
    interiors = frame[frame["orbit_size"].eq(4)].sort_values(
        ["selection_priority", "fundamental_representative_id"], kind="stable"
    )
    if len(zero) != 1:
        raise ValueError("exactly one zero representative is required")
    target = int(target_expanded_rows)
    if target < 3 or target % 2 == 0:
        raise ValueError("retry12 expanded target must be odd")
    mandatory_count = int(seams["mandatory_seam_representative"].astype(bool).sum())
    seam_count = max(1, mandatory_count)
    if seam_count % 2 == 0:
        seam_count += 1
    while seam_count <= len(seams) and 1 + 2 * seam_count > target:
        seam_count -= 2
    if seam_count < mandatory_count or seam_count < 1:
        raise ValueError("mandatory seam representatives exceed orbit budget")
    remaining = target - 1 - 2 * seam_count
    if remaining < 0 or remaining % 4:
        raise ValueError("orbit parity contract cannot be satisfied")
    interior_count = remaining // 4
    if interior_count > len(interiors):
        raise ValueError("insufficient interior representatives for exact orbit budget")
    selected = pd.concat([zero, seams.head(seam_count), interiors.head(interior_count)], ignore_index=True)
    if int(selected["orbit_size"].sum()) != target:
        raise AssertionError("orbit selector failed exact expanded target")
    return selected


def seam_continuity_metrics(
    eligible: pd.DataFrame,
    retained: pd.DataFrame,
    *,
    seam_class: str,
    maximum_edge_mm: float,
) -> dict[str, Any]:
    expected = eligible[eligible["seam_class"].eq(seam_class)].copy()
    served = retained[retained["seam_class"].eq(seam_class)].copy()
    parent_total = int(expected["quotient_parent_id"].nunique())
    parent_served = int(served["quotient_parent_id"].nunique())
    shells = sorted(expected["radial_shell_id"].astype(int).unique())
    served_shells = set(served["radial_shell_id"].astype(int))
    longest = current = 0
    for shell in shells:
        if shell in served_shells:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    maximum_verified_edge = math.inf
    if "source_edge_length_mm" in served and len(served):
        finite_edges = served["source_edge_length_mm"].to_numpy(float)
        finite_edges = finite_edges[np.isfinite(finite_edges)]
        maximum_verified_edge = float(np.max(finite_edges)) if len(finite_edges) else math.inf
    elif len(served) >= 2:
        ordered = served.sort_values(["zero_radius_mm", "fundamental_representative_id"], kind="stable")
        distance = np.linalg.norm(np.diff(ordered.loc[:, XYZ_COLUMNS].to_numpy(float), axis=0), axis=1) * 1000.0
        maximum_verified_edge = float(np.max(distance)) if len(distance) else 0.0
    return {
        "seam_class": seam_class,
        "eligible_parent_count": parent_total,
        "served_parent_count": parent_served,
        "served_parent_fraction": parent_served / parent_total if parent_total else 0.0,
        "eligible_shell_count": len(shells),
        "served_shell_count": len(served_shells.intersection(shells)),
        "served_shell_fraction": len(served_shells.intersection(shells)) / len(shells) if shells else 0.0,
        "maximum_consecutive_empty_shells": longest,
        "maximum_verified_edge_length_mm": maximum_verified_edge,
        "verified_edge_length_pass": bool(maximum_verified_edge <= float(maximum_edge_mm) + 1.0e-12),
    }


def corrected_trajectory_metrics(raw_beta: np.ndarray, corrected_beta: np.ndarray) -> dict[str, float]:
    raw = np.asarray(raw_beta, dtype=float).reshape(-1, 6)
    corrected = np.asarray(corrected_beta, dtype=float).reshape(-1, 6)
    if len(raw) != len(corrected):
        raise ValueError("raw and corrected trajectories must have equal length")
    raw_steps = np.max(np.abs(np.diff(raw, axis=0)), axis=1) * 180.0 / np.pi
    corrected_steps = np.max(np.abs(np.diff(corrected, axis=0)), axis=1) * 180.0 / np.pi
    correction = np.max(np.abs(corrected - raw), axis=1) * 180.0 / np.pi
    return {
        "raw_step_gt7_rate": float(np.mean(raw_steps > 7.0)) if len(raw_steps) else 0.0,
        "dls2_corrected_step_gt7_rate": float(np.mean(corrected_steps > 7.0)) if len(corrected_steps) else 0.0,
        "correction_magnitude_p95_deg": float(np.percentile(correction, 95)) if len(correction) else 0.0,
    }
