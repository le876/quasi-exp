#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from fk_dh_numpy import fk_dh_batch  # noqa: E402
from true_ellipse_radial_bundle_engine import joint_margin_barrier_residual  # noqa: E402


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
XYZ_COLS = ["x_m", "y_m", "z_m"]
TARGET_XYZ_COLS = ["x_target_m", "y_target_m", "z_target_m"]


@dataclass(frozen=True)
class IKSolution:
    beta_rad: np.ndarray
    xyz_m: np.ndarray
    residual_mm: float
    success: bool
    nfev: int
    cost: float
    seed_rank: int


DEFAULT_TRAJECTORY_STAGES: tuple[dict[str, float | str], ...] = (
    {
        "name": "stage_a",
        "lambda_velocity": 0.1,
        "lambda_acceleration": 0.0,
        "lambda_anchor": 0.1,
        "lambda_posture": 0.0,
    },
    {
        "name": "stage_b",
        "lambda_velocity": 1.0,
        "lambda_acceleration": 0.25,
        "lambda_anchor": 0.01,
        "lambda_posture": 0.01,
    },
    {
        "name": "stage_c",
        "lambda_velocity": 2.0,
        "lambda_acceleration": 0.5,
        "lambda_anchor": 0.0,
        "lambda_posture": 0.05,
    },
)


def beta_bounds_rad(domain: str = "current") -> np.ndarray:
    domain = str(domain).strip().lower()
    if domain == "current":
        deg = np.asarray([[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-15, 15], [-15, 15]], dtype=float)
    elif domain == "relaxed_proximal":
        deg = np.asarray([[-7.5, 7.5], [-7.5, 7.5], [-10, 10], [-10, 10], [-15, 15], [-15, 15]], dtype=float)
    else:
        raise ValueError(f"unsupported beta domain: {domain}")
    return np.deg2rad(deg)


def theta_from_beta_batch(beta_rad: np.ndarray, *, theta_sign: float = -1.0) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    theta = np.empty((len(beta), 30), dtype=float)
    for section, (odd_col, even_col) in enumerate(((0, 1), (2, 3), (4, 5))):
        start = section * 10
        theta[:, start : start + 10 : 2] = beta[:, odd_col : odd_col + 1]
        theta[:, start + 1 : start + 10 : 2] = beta[:, even_col : even_col + 1]
    return theta * float(theta_sign)


def fk_from_beta_batch(
    beta_rad: np.ndarray,
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float = -1.0,
) -> np.ndarray:
    theta = theta_from_beta_batch(beta_rad, theta_sign=theta_sign)
    return fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)


def sobol_beta_samples(n: int, *, bounds: np.ndarray, seed: int) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.zeros((0, 6), dtype=float)
    bounds = np.asarray(bounds, dtype=float).reshape(6, 2)
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=6, scramble=True, seed=int(seed))
        m = int(np.ceil(np.log2(max(1, n))))
        unit = np.asarray(sampler.random_base2(m), dtype=float)[:n]
    except Exception:  # noqa: BLE001
        rng = np.random.default_rng(int(seed))
        unit = rng.random((n, 6))
    return bounds[:, 0][None, :] + unit * (bounds[:, 1] - bounds[:, 0])[None, :]


def make_reachability_pool(
    *,
    n: int,
    domain: str,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    seed: int,
    source_domain: str | None = None,
) -> pd.DataFrame:
    bounds = beta_bounds_rad(domain)
    beta = sobol_beta_samples(int(n), bounds=bounds, seed=int(seed))
    theta = theta_from_beta_batch(beta, theta_sign=theta_sign)
    xyz = fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    out = pd.DataFrame({"sample_id": np.arange(len(beta), dtype=np.int64)})
    out["source_domain"] = str(source_domain or domain)
    for i, col in enumerate(BETA_COLS):
        out[col] = beta[:, i]
    for i, col in enumerate(THETA_COLS):
        out[col] = theta[:, i]
    for i, col in enumerate(XYZ_COLS):
        out[col] = xyz[:, i]
    return out


def make_axis_phase_ellipse(
    *,
    candidate_id: str,
    center: tuple[float, float, float],
    amp_xy_mm: float,
    phase_y_rad: float,
    phase_z_rad: float,
    n_points: int,
    include_endpoint: bool = False,
) -> pd.DataFrame:
    n = int(n_points)
    t = np.linspace(0.0, 2.0 * math.pi, n + (1 if include_endpoint else 0), endpoint=include_endpoint, dtype=float)
    amp_xy = float(amp_xy_mm) / 1000.0
    amp_z = 1.5 * amp_xy
    cx, cy, cz = (float(v) for v in center)
    xyz = np.column_stack(
        [
            cx + amp_xy * np.sin(t),
            cy + amp_xy * np.sin(t + float(phase_y_rad)),
            cz + amp_z * np.sin(t + float(phase_z_rad)),
        ]
    )
    out = pd.DataFrame(
        {
            "candidate_id": str(candidate_id),
            "ellipse_id": str(candidate_id),
            "angle_idx": np.arange(len(t), dtype=np.int64),
            "angle_rad": t,
            "center_x_m": cx,
            "center_y_m": cy,
            "center_z_m": cz,
            "amp_xy_mm": float(amp_xy_mm),
            "amp_z_mm": 1.5 * float(amp_xy_mm),
            "phase_y_rad": float(phase_y_rad),
            "phase_z_rad": float(phase_z_rad),
        }
    )
    for i, col in enumerate(TARGET_XYZ_COLS):
        out[col] = xyz[:, i]
    return out


def trajectory_geometry_metrics(xyz: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if len(pts) < 3:
        return {"rank2_gate_pass": False}
    centered = pts - pts.mean(axis=0, keepdims=True)
    _u, s, _vh = np.linalg.svd(centered, full_matrices=False)
    s = np.pad(s, (0, max(0, 3 - len(s))), constant_values=0.0)
    svd1 = max(float(s[0]), 1.0e-15)
    ranges = np.ptp(pts, axis=0)
    amp_xy_mean = max(float(np.mean(ranges[:2]) / 2.0), 1.0e-15)
    closed_gap = float(np.linalg.norm(pts[0] - pts[-1])) if np.allclose(pts[0], pts[-1]) else 0.0
    area = float(0.5 * np.linalg.norm(np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0)))
    out = {
        "svd1_m": float(s[0]),
        "svd2_m": float(s[1]),
        "svd3_m": float(s[2]),
        "svd2_over_svd1": float(s[1] / svd1),
        "svd3_over_svd1": float(s[2] / svd1),
        "closed_loop_gap_m": closed_gap,
        "amp_x_m": float(ranges[0] / 2.0),
        "amp_y_m": float(ranges[1] / 2.0),
        "amp_z_m": float(ranges[2] / 2.0),
        "amp_z_over_xy_mean": float((ranges[2] / 2.0) / amp_xy_mean),
        "polygon_area_m2": area,
    }
    out["rank2_gate_pass"] = bool(out["svd2_over_svd1"] > 0.3 and out["svd3_over_svd1"] < 1.0e-3 and area > 1.0e-8)
    return out


def support_summary_for_targets(
    targets: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    pool_label: str,
    tube_radius_mm: float = 15.0,
    beta_pair_radius_mm: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_xyz = targets[TARGET_XYZ_COLS].to_numpy(dtype=float)
    pool_xyz = pool[XYZ_COLS].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=1).fit(pool_xyz)
    dist, idx = nn.kneighbors(target_xyz)
    radius_nn = NearestNeighbors().fit(pool_xyz)
    tube_indices = radius_nn.radius_neighbors(target_xyz, radius=float(tube_radius_mm) / 1000.0, return_distance=False)
    counts = np.asarray([len(v) for v in tube_indices], dtype=float)
    per = targets[["candidate_id", "angle_idx", "angle_rad", *TARGET_XYZ_COLS]].copy()
    per["pool_label"] = str(pool_label)
    per["true_nn_mm"] = dist[:, 0] * 1000.0
    per["nearest_pool_row"] = idx[:, 0].astype(np.int64)
    per["true_tube_count_15mm"] = counts
    beta_p95 = float("nan")
    if all(c in pool.columns for c in BETA_COLS):
        unique = np.unique(np.concatenate([v for v in tube_indices if len(v) > 0])) if np.any(counts > 0) else np.empty(0, dtype=np.int64)
        if len(unique) >= 2:
            xyz = pool.iloc[unique][XYZ_COLS].to_numpy(dtype=float)
            beta = pool.iloc[unique][BETA_COLS].to_numpy(dtype=float)
            k = min(16, len(unique))
            local_nn = NearestNeighbors(n_neighbors=k).fit(xyz)
            d2, i2 = local_nn.kneighbors(xyz)
            vals: list[float] = []
            max_m = float(beta_pair_radius_mm) / 1000.0
            for i in range(len(xyz)):
                mask = (d2[i] > 1.0e-12) & (d2[i] <= max_m)
                if np.any(mask):
                    diff = beta[i2[i][mask]] - beta[i]
                    vals.extend((np.sqrt(np.mean(np.square(diff), axis=1)) * 180.0 / math.pi).tolist())
            if vals:
                beta_p95 = float(np.percentile(vals, 95))
    rows = []
    for cid, part in per.groupby("candidate_id", sort=False):
        nn_mm = part["true_nn_mm"].to_numpy(dtype=float)
        cts = part["true_tube_count_15mm"].to_numpy(dtype=float)
        rows.append(
            {
                "pool_label": str(pool_label),
                "candidate_id": cid,
                "true_nn_p95_mm": float(np.percentile(nn_mm, 95)),
                "true_nn_max_mm": float(np.max(nn_mm)),
                "true_tube_count_p10": float(np.percentile(cts, 10)),
                "true_tube_beta_rms_p95_deg": beta_p95,
                "true_support_gate_pass": bool(np.percentile(nn_mm, 95) <= 8.0 and np.percentile(cts, 10) >= 16.0),
            }
        )
    return pd.DataFrame(rows), per


def canonical_posture_penalty(beta_rad: np.ndarray, bounds: np.ndarray | None = None) -> float:
    beta = np.asarray(beta_rad, dtype=float).reshape(6)
    if bounds is None:
        bounds = beta_bounds_rad("current")
    scale = np.maximum(np.max(np.abs(bounds), axis=1), 1.0e-12)
    norm = beta / scale
    return float(4.0 * np.mean(norm[:2] ** 2) + 2.0 * np.mean(norm[2:4] ** 2) + np.mean(norm[4:] ** 2))


def numerical_jacobian_beta(
    beta_rad: np.ndarray,
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    eps_rad: float = 1.0e-4,
) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float).reshape(6)
    jac = np.empty((3, 6), dtype=float)
    for j in range(6):
        step = np.zeros(6, dtype=float)
        step[j] = float(eps_rad)
        p_plus = fk_from_beta_batch((beta + step).reshape(1, 6), lengths_m=lengths_m, p_end_local_m=p_end_local_m, theta_sign=theta_sign)[0]
        p_minus = fk_from_beta_batch((beta - step).reshape(1, 6), lengths_m=lengths_m, p_end_local_m=p_end_local_m, theta_sign=theta_sign)[0]
        jac[:, j] = (p_plus - p_minus) / (2.0 * float(eps_rad))
    return jac


def jacobian_metrics(jac: np.ndarray) -> dict[str, float]:
    s = np.linalg.svd(np.asarray(jac, dtype=float).reshape(3, 6), compute_uv=False)
    s = np.pad(s, (0, max(0, 3 - len(s))), constant_values=0.0)
    sigma3 = max(float(s[2]), 1.0e-15)
    return {"sigma1_m": float(s[0]), "sigma2_m": float(s[1]), "sigma3_m": float(s[2]), "kappa": float(s[0] / sigma3)}


def solve_beta_ik_many(
    target_xyz_m: np.ndarray,
    *,
    init_betas: np.ndarray,
    bounds: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    max_nfev: int = 100,
    lambda_limit: float = 0.01,
    center_beta: np.ndarray | None = None,
    lambda_center: float = 0.0,
    center_each_seed: bool = False,
) -> list[IKSolution]:
    target = np.asarray(target_xyz_m, dtype=float).reshape(3)
    init = np.asarray(init_betas, dtype=float).reshape(-1, 6)
    if len(init) == 0:
        raise ValueError("init_betas must contain at least one seed")
    bounds = np.asarray(bounds, dtype=float).reshape(6, 2)
    lo, hi = bounds[:, 0], bounds[:, 1]
    scale = np.maximum(hi - lo, 1.0e-12)
    fixed_center = np.zeros(6, dtype=float) if center_beta is None else np.asarray(center_beta, dtype=float).reshape(6)
    solutions: list[IKSolution] = []

    for seed_rank, x0 in enumerate(init):
        x0 = np.clip(np.asarray(x0, dtype=float).reshape(6), lo, hi)
        center = x0 if center_each_seed else fixed_center

        def residual(x: np.ndarray) -> np.ndarray:
            xyz = fk_from_beta_batch(
                x.reshape(1, 6),
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
            )[0]
            parts = [((xyz - target) / 0.002).astype(float)]
            if lambda_limit > 0.0:
                mid = 0.5 * (lo + hi)
                parts.append(math.sqrt(float(lambda_limit)) * ((x - mid) / scale))
            if lambda_center > 0.0:
                parts.append(math.sqrt(float(lambda_center)) * ((x - center) / np.deg2rad(1.0)))
            return np.concatenate(parts)

        res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=int(max_nfev), xtol=1.0e-9, ftol=1.0e-9, gtol=1.0e-9)
        beta = np.asarray(res.x, dtype=float)
        xyz = fk_from_beta_batch(beta.reshape(1, 6), lengths_m=lengths_m, p_end_local_m=p_end_local_m, theta_sign=theta_sign)[0]
        err_mm = float(np.linalg.norm(xyz - target) * 1000.0)
        sol = IKSolution(beta_rad=beta, xyz_m=xyz, residual_mm=err_mm, success=bool(res.success and err_mm <= 5.0), nfev=int(res.nfev), cost=float(res.cost), seed_rank=int(seed_rank))
        solutions.append(sol)
    return solutions


def solve_beta_ik(
    target_xyz_m: np.ndarray,
    *,
    init_betas: np.ndarray,
    bounds: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    max_nfev: int = 100,
    lambda_limit: float = 0.01,
    center_beta: np.ndarray | None = None,
    lambda_center: float = 0.0,
) -> IKSolution:
    solutions = solve_beta_ik_many(
        target_xyz_m,
        init_betas=init_betas,
        bounds=bounds,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        max_nfev=max_nfev,
        lambda_limit=lambda_limit,
        center_beta=center_beta,
        lambda_center=lambda_center,
    )
    return min(solutions, key=lambda sol: (sol.residual_mm, sol.cost, sol.seed_rank))


def generate_nullspace_seeds(
    beta_rad: np.ndarray,
    jac: np.ndarray,
    *,
    bounds: np.ndarray,
    scales_deg: Iterable[float],
    seeds_per_scale: int = 6,
    seed: int = 0,
) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float).reshape(6)
    j = np.asarray(jac, dtype=float).reshape(3, 6)
    limits = np.asarray(bounds, dtype=float).reshape(6, 2)
    _u, singular, vh = np.linalg.svd(j, full_matrices=True)
    tol = max(j.shape) * np.finfo(float).eps * max(float(singular[0]) if len(singular) else 0.0, 1.0)
    rank = int(np.sum(singular > tol))
    null_basis = vh[rank:, :]
    if len(null_basis) == 0 or int(seeds_per_scale) <= 0:
        return np.empty((0, 6), dtype=float)

    rng = np.random.default_rng(int(seed))
    directions: list[np.ndarray] = []
    for i in range(int(seeds_per_scale)):
        if i < 2 * len(null_basis):
            basis = null_basis[i // 2]
            direction = basis if i % 2 == 0 else -basis
        else:
            coeff = rng.normal(size=len(null_basis))
            direction = coeff @ null_basis
        rms = float(np.sqrt(np.mean(np.square(direction))))
        if rms <= 1.0e-15:
            continue
        directions.append(direction / rms)

    seeds: list[np.ndarray] = []
    for scale_deg in scales_deg:
        scale_rad = float(np.deg2rad(float(scale_deg)))
        for direction in directions:
            seeds.append(np.clip(beta + scale_rad * direction, limits[:, 0], limits[:, 1]))
    return np.asarray(seeds, dtype=float).reshape(-1, 6)


def cluster_beta_candidates(candidates: pd.DataFrame, *, beta_rms_threshold_deg: float = 1.0, max_clusters: int = 16) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    work = candidates.sort_values("xyz_residual_mm", ascending=True).reset_index(drop=True)
    kept: list[pd.Series] = []
    thresh = float(beta_rms_threshold_deg)
    for _, row in work.iterrows():
        beta = row[BETA_COLS].to_numpy(dtype=float)
        duplicate = False
        for old in kept:
            old_beta = old[BETA_COLS].to_numpy(dtype=float)
            rms = float(np.sqrt(np.mean(np.square(beta - old_beta))) * 180.0 / math.pi)
            if rms < thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
        if len(kept) >= int(max_clusters):
            break
    out = pd.DataFrame(kept).reset_index(drop=True)
    out["branch_cluster_id"] = np.arange(len(out), dtype=np.int64)
    return out


def beta_rms_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))) * 180.0 / math.pi)


def _trajectory_targets_frame(targets: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(targets, pd.DataFrame):
        frame = targets.copy().reset_index(drop=True)
        if all(col in frame.columns for col in TARGET_XYZ_COLS):
            return frame
        if all(col in frame.columns for col in XYZ_COLS):
            for target_col, xyz_col in zip(TARGET_XYZ_COLS, XYZ_COLS):
                frame[target_col] = frame[xyz_col].to_numpy(dtype=float)
            return frame
        raise ValueError(f"targets must contain {TARGET_XYZ_COLS} or {XYZ_COLS}")
    xyz = np.asarray(targets, dtype=float).reshape(-1, 3)
    n = len(xyz)
    frame = pd.DataFrame(
        {
            "angle_idx": np.arange(n, dtype=np.int64),
            "angle_rad": np.linspace(0.0, 2.0 * math.pi, n, endpoint=False),
        }
    )
    for j, col in enumerate(TARGET_XYZ_COLS):
        frame[col] = xyz[:, j]
    return frame


def _solution_record(
    target_row: pd.Series,
    solution: IKSolution,
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
) -> dict[str, Any]:
    record = target_row.to_dict()
    record.update(
        {
            "xyz_residual_mm": float(solution.residual_mm),
            "ik_success": bool(solution.success),
            "inverse_nfev": int(solution.nfev),
            "seed_rank": int(solution.seed_rank),
        }
    )
    for j, col in enumerate(BETA_COLS):
        record[col] = float(solution.beta_rad[j])
    theta = theta_from_beta_batch(solution.beta_rad.reshape(1, 6), theta_sign=theta_sign)[0]
    for j, col in enumerate(THETA_COLS):
        record[col] = float(theta[j])
    for j, col in enumerate(XYZ_COLS):
        record[col] = float(solution.xyz_m[j])
    jac = numerical_jacobian_beta(
        solution.beta_rad,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
    )
    record.update(jacobian_metrics(jac))
    record["canonical_posture_cost"] = canonical_posture_penalty(solution.beta_rad)
    return record


def continuation_lift(
    *,
    targets: pd.DataFrame | np.ndarray,
    start_beta: np.ndarray,
    bounds: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    direction: str = "forward",
    method: str = "warm",
    lambda_center: float = 1.0e-3,
    lambda_limit: float = 0.0,
    max_nfev: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _trajectory_targets_frame(targets)
    n = len(frame)
    if n == 0:
        return frame.copy(), {"success": False, "reason": "empty_targets", "rows": 0}
    direction = str(direction).strip().lower()
    method = str(method).strip().lower()
    if direction not in {"forward", "reverse"}:
        raise ValueError(f"unsupported continuation direction: {direction}")
    if method not in {"warm", "predictive"}:
        raise ValueError(f"unsupported continuation method: {method}")
    traversal = list(range(n)) if direction == "forward" else [0, *range(n - 1, 0, -1)]
    history: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for traversal_rank, row_idx in enumerate(traversal):
        target_row = frame.iloc[int(row_idx)]
        if not history:
            prediction = np.asarray(start_beta, dtype=float).reshape(6)
            seed_matrix = prediction.reshape(1, 6)
        else:
            warm = history[-1]
            prediction = warm
            if method == "predictive" and len(history) >= 2:
                prediction = history[-1] + (history[-1] - history[-2])
            prediction = np.clip(prediction, np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1])
            seed_matrix = np.vstack([prediction, warm])
        solutions = solve_beta_ik_many(
            target_row[TARGET_XYZ_COLS].to_numpy(dtype=float),
            init_betas=seed_matrix,
            bounds=bounds,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
            max_nfev=max_nfev,
            lambda_limit=lambda_limit,
            center_beta=prediction,
            lambda_center=lambda_center,
        )
        tracking_ok = [sol for sol in solutions if sol.residual_mm <= 2.0]
        if tracking_ok:
            chosen = min(
                tracking_ok,
                key=lambda sol: (beta_rms_deg(sol.beta_rad, prediction), sol.residual_mm, sol.cost),
            )
        else:
            chosen = min(solutions, key=lambda sol: (sol.residual_mm, beta_rms_deg(sol.beta_rad, prediction)))
        history.append(chosen.beta_rad)
        record = _solution_record(
            target_row,
            chosen,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
        )
        record.update(
            {
                "traversal_rank": int(traversal_rank),
                "continuation_direction": direction,
                "continuation_method": method,
                "lambda_center": float(lambda_center),
            }
        )
        records.append(record)

    result = pd.DataFrame(records).sort_values("angle_idx").reset_index(drop=True)
    report = smoothness_report(result)
    report.update(
        {
            "success": bool(report.get("residual_max_mm", np.inf) <= 5.0),
            "direction": direction,
            "method": method,
            "lambda_center": float(lambda_center),
            "canonical_posture_mean": float(result["canonical_posture_cost"].mean()),
        }
    )
    return result, report


def branch_reproducibility_report(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    threshold_deg: float = 1.0,
) -> dict[str, Any]:
    if left.empty or right.empty:
        return {"rows": 0, "branch_diff_p95_deg": float("inf"), "reproducible": False}
    left_sorted = left.sort_values("angle_idx").reset_index(drop=True)
    right_sorted = right.sort_values("angle_idx").reset_index(drop=True)
    if left_sorted["angle_idx"].tolist() != right_sorted["angle_idx"].tolist():
        raise ValueError("branch comparison requires identical angle_idx values")
    a = left_sorted[BETA_COLS].to_numpy(dtype=float)
    b = right_sorted[BETA_COLS].to_numpy(dtype=float)
    diff = np.sqrt(np.mean(np.square(a - b), axis=1)) * 180.0 / math.pi
    return {
        "rows": int(len(diff)),
        "branch_diff_mean_deg": float(np.mean(diff)),
        "branch_diff_p95_deg": float(np.percentile(diff, 95)),
        "branch_diff_max_deg": float(np.max(diff)),
        "reproducible": bool(np.percentile(diff, 95) <= float(threshold_deg)),
    }


def branch_hash(df: pd.DataFrame, *, decimals: int = 10) -> str:
    ordered = df.sort_values("angle_idx")[BETA_COLS].to_numpy(dtype=float)
    payload = np.round(ordered, decimals=int(decimals)).tobytes()
    return hashlib.sha256(payload).hexdigest()


def link_cyclic_branch(
    candidates: pd.DataFrame,
    *,
    max_edge_deg: float = 3.0,
    closure_weight: float = 2.0,
    lambda_posture: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if candidates.empty:
        return candidates.copy(), {"success": False, "reason": "empty_candidates"}
    layers = [g.reset_index(drop=True) for _, g in candidates.groupby("angle_idx", sort=True)]
    if not layers:
        return candidates.copy(), {"success": False, "reason": "no_layers"}
    betas = [g[BETA_COLS].to_numpy(dtype=float) for g in layers]
    unary = [
        (g["xyz_residual_mm"].to_numpy(dtype=float) / 2.0) ** 2
        + float(lambda_posture) * np.asarray([canonical_posture_penalty(b) for b in betas[i]], dtype=float)
        for i, g in enumerate(layers)
    ]
    best_cost = float("inf")
    best_path: list[int] | None = None
    n0 = len(layers[0])
    for start in range(n0):
        costs = np.full(len(layers[0]), np.inf, dtype=float)
        costs[start] = unary[0][start]
        parents: list[np.ndarray] = []
        for i in range(1, len(layers)):
            prev_beta = betas[i - 1]
            cur_beta = betas[i]
            new_costs = np.full(len(cur_beta), np.inf, dtype=float)
            parent = np.full(len(cur_beta), -1, dtype=np.int64)
            for j in range(len(cur_beta)):
                d = np.asarray([beta_rms_deg(pb, cur_beta[j]) for pb in prev_beta], dtype=float)
                edge = (d / 1.0) ** 2
                edge[d > float(max_edge_deg)] = np.inf
                vals = costs + edge + unary[i][j]
                p = int(np.argmin(vals))
                if np.isfinite(vals[p]):
                    new_costs[j] = vals[p]
                    parent[j] = p
            parents.append(parent)
            costs = new_costs
        for end in range(len(layers[-1])):
            if not np.isfinite(costs[end]):
                continue
            seam = beta_rms_deg(betas[-1][end], betas[0][start])
            if seam > float(max_edge_deg):
                continue
            total = float(costs[end] + float(closure_weight) * (seam / 1.0) ** 2)
            if total < best_cost:
                path = [end]
                cur = end
                for parent in reversed(parents):
                    cur = int(parent[cur])
                    path.append(cur)
                best_path = list(reversed(path))
                best_cost = total
    if best_path is None:
        return pd.DataFrame(), {"success": False, "reason": "no_closed_path"}
    selected = pd.concat([layers[i].iloc[[best_path[i]]] for i in range(len(layers))], ignore_index=True)
    beta = selected[BETA_COLS].to_numpy(dtype=float)
    d = np.asarray([beta_rms_deg(beta[i], beta[(i + 1) % len(beta)]) for i in range(len(beta))], dtype=float)
    report = {
        "success": True,
        "cost": best_cost,
        "rows": int(len(selected)),
        "delta_beta_rms_p95_deg": float(np.percentile(d, 95)),
        "delta_beta_rms_max_deg": float(np.max(d)),
        "seam_beta_rms_deg": float(d[-1]),
        "reconfiguration_count": int(np.sum(d > float(max_edge_deg))),
        "zero_reconfiguration": bool(np.all(d <= float(max_edge_deg))),
    }
    return selected, report


def _pairwise_beta_rms_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=float).reshape(-1, 6)
    b = np.asarray(right, dtype=float).reshape(-1, 6)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.mean(np.square(diff), axis=2)) * 180.0 / math.pi


def link_cyclic_branch_soft(
    candidates: pd.DataFrame,
    *,
    lambda_velocity: float = 1.0,
    closure_weight: float = 5.0,
    lambda_posture: float = 0.05,
    lambda_kappa: float = 0.01,
    residual_scale_mm: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if candidates.empty:
        return candidates.copy(), {"success": False, "reason": "empty_candidates"}
    layers = [part.sort_values("xyz_residual_mm").reset_index(drop=True) for _, part in candidates.groupby("angle_idx", sort=True)]
    if not layers:
        return candidates.copy(), {"success": False, "reason": "no_layers"}
    betas = [part[BETA_COLS].to_numpy(dtype=float) for part in layers]
    unary: list[np.ndarray] = []
    for part, layer_beta in zip(layers, betas):
        residual = part["xyz_residual_mm"].to_numpy(dtype=float) / max(float(residual_scale_mm), 1.0e-12)
        posture = np.asarray([canonical_posture_penalty(beta) for beta in layer_beta], dtype=float)
        if "kappa" in part.columns:
            kappa = np.nan_to_num(part["kappa"].to_numpy(dtype=float), nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
        else:
            kappa = np.zeros(len(part), dtype=float)
        unary.append(np.square(residual) + float(lambda_posture) * posture + float(lambda_kappa) * np.log1p(np.maximum(kappa, 0.0)))

    transitions = [
        float(lambda_velocity) * np.square(_pairwise_beta_rms_deg(betas[i - 1], betas[i]))
        for i in range(1, len(layers))
    ]
    best_cost = float("inf")
    best_path: list[int] | None = None
    for start in range(len(layers[0])):
        costs = np.full(len(layers[0]), np.inf, dtype=float)
        costs[start] = unary[0][start]
        parents: list[np.ndarray] = []
        for layer_idx in range(1, len(layers)):
            values = costs[:, None] + transitions[layer_idx - 1]
            parent = np.argmin(values, axis=0).astype(np.int64)
            costs = values[parent, np.arange(values.shape[1])] + unary[layer_idx]
            parents.append(parent)
        seam = _pairwise_beta_rms_deg(betas[-1], betas[0][start : start + 1])[:, 0]
        totals = costs + float(closure_weight) * np.square(seam)
        end = int(np.argmin(totals))
        total = float(totals[end])
        if total >= best_cost:
            continue
        path = [end]
        current = end
        for parent in reversed(parents):
            current = int(parent[current])
            path.append(current)
        best_path = list(reversed(path))
        best_cost = total

    if best_path is None:
        return pd.DataFrame(), {"success": False, "reason": "no_closed_path"}
    selected = pd.concat([layers[i].iloc[[best_path[i]]] for i in range(len(layers))], ignore_index=True)
    report = smoothness_report(selected)
    report.update(
        {
            "success": True,
            "cost": float(best_cost),
            "lambda_velocity": float(lambda_velocity),
            "closure_weight": float(closure_weight),
            "lambda_posture": float(lambda_posture),
            "lambda_kappa": float(lambda_kappa),
            "delta_beta_rms_p95_deg": float(report["delta_beta_p95_deg"]),
            "delta_beta_rms_max_deg": float(report["delta_beta_max_deg"]),
        }
    )
    return selected, report


def smoothness_report(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or not all(c in df.columns for c in BETA_COLS):
        return {"rows": int(len(df))}
    work = df.sort_values("angle_idx").reset_index(drop=True) if "angle_idx" in df.columns else df.reset_index(drop=True)
    beta = work[BETA_COLS].to_numpy(dtype=float)
    if len(beta) <= 1:
        d = np.asarray([0.0])
        dd = np.asarray([0.0])
    else:
        d = np.asarray([beta_rms_deg(beta[i], beta[(i + 1) % len(beta)]) for i in range(len(beta))], dtype=float)
        dd_beta = np.roll(beta, -1, axis=0) - 2.0 * beta + np.roll(beta, 1, axis=0)
        dd = np.sqrt(np.mean(np.square(dd_beta), axis=1)) * 180.0 / math.pi
    out = {
        "rows": int(len(df)),
        "delta_beta_p95_deg": float(np.percentile(d, 95)),
        "delta_beta_max_deg": float(np.max(d)),
        "delta2_beta_p95_deg": float(np.percentile(dd, 95)),
        "seam_beta_rms_deg": float(d[-1]),
        "canonical_posture_mean": float(np.mean([canonical_posture_penalty(row) for row in beta])),
    }
    phase_step = 2.0 * math.pi / max(len(beta), 1)
    if "angle_rad" in work.columns and len(work) > 1:
        phase = np.unwrap(work["angle_rad"].to_numpy(dtype=float))
        nonzero = np.diff(phase)
        nonzero = np.abs(nonzero[np.abs(nonzero) > 1.0e-12])
        if len(nonzero):
            phase_step = float(np.median(nonzero))
    out["phase_step_rad"] = float(phase_step)
    out["beta_change_per_phase_p95_deg_per_rad"] = float(np.percentile(d, 95) / max(phase_step, 1.0e-12))
    out["beta_change_per_phase_max_deg_per_rad"] = float(np.max(d) / max(phase_step, 1.0e-12))
    if "xyz_residual_mm" in work.columns:
        out["residual_p95_mm"] = float(np.percentile(work["xyz_residual_mm"].to_numpy(dtype=float), 95))
        out["residual_max_mm"] = float(np.max(work["xyz_residual_mm"].to_numpy(dtype=float)))
    if "kappa" in work.columns:
        out["kappa_p95"] = float(np.percentile(work["kappa"].to_numpy(dtype=float), 95))
    if "sigma3_m" in work.columns:
        out["sigma3_p05_m"] = float(np.percentile(work["sigma3_m"].to_numpy(dtype=float), 5))
    return out


def cyclic_trajectory_residual(
    flat_beta: np.ndarray,
    *,
    targets_xyz: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    anchor_beta: np.ndarray | None = None,
    bounds: np.ndarray | None = None,
    lambda_velocity: float = 1.0,
    lambda_acceleration: float = 0.0,
    lambda_anchor: float = 0.0,
    lambda_posture: float = 0.0,
    lambda_margin: float = 0.0,
    soft_margin_deg: float = 0.25,
    tracking_scale_m: float = 0.001,
    velocity_scale_rad: float = math.pi / 180.0,
    acceleration_scale_rad: float = math.pi / 720.0,
) -> np.ndarray:
    targets = np.asarray(targets_xyz, dtype=float).reshape(-1, 3)
    beta = np.asarray(flat_beta, dtype=float).reshape(len(targets), 6)
    xyz = fk_from_beta_batch(
        beta,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
    )
    parts = [((xyz - targets) / max(float(tracking_scale_m), 1.0e-12)).reshape(-1)]
    velocity = np.roll(beta, -1, axis=0) - beta
    parts.append((math.sqrt(max(float(lambda_velocity), 0.0)) * velocity / max(float(velocity_scale_rad), 1.0e-12)).reshape(-1))
    if lambda_acceleration > 0.0:
        acceleration = np.roll(beta, -1, axis=0) - 2.0 * beta + np.roll(beta, 1, axis=0)
        parts.append((math.sqrt(float(lambda_acceleration)) * acceleration / max(float(acceleration_scale_rad), 1.0e-12)).reshape(-1))
    if lambda_anchor > 0.0:
        if anchor_beta is None:
            raise ValueError("anchor_beta is required when lambda_anchor > 0")
        anchor = np.asarray(anchor_beta, dtype=float).reshape(beta.shape)
        parts.append((math.sqrt(float(lambda_anchor)) * (beta - anchor) / max(float(velocity_scale_rad), 1.0e-12)).reshape(-1))
    if lambda_posture > 0.0:
        limits = beta_bounds_rad("current") if bounds is None else np.asarray(bounds, dtype=float).reshape(6, 2)
        scale = np.maximum(np.max(np.abs(limits), axis=1), 1.0e-12)
        coeff = np.sqrt(np.asarray([2.0, 2.0, 1.0, 1.0, 0.5, 0.5], dtype=float))
        parts.append((math.sqrt(float(lambda_posture)) * (beta / scale[None, :]) * coeff[None, :]).reshape(-1))
    if lambda_margin > 0.0:
        limits = beta_bounds_rad("current") if bounds is None else np.asarray(bounds, dtype=float).reshape(6, 2)
        parts.append(
            math.sqrt(float(lambda_margin))
            * joint_margin_barrier_residual(
                beta,
                bounds_rad=limits,
                soft_margin_deg=float(soft_margin_deg),
            )
        )
    return np.concatenate(parts)


def cyclic_trajectory_jac_sparsity(
    *,
    n_points: int,
    include_acceleration: bool,
    include_anchor: bool,
    include_posture: bool,
    include_margin: bool = False,
) -> sparse.csr_matrix:
    n = int(n_points)
    if n <= 0:
        return sparse.csr_matrix((0, 0), dtype=np.int8)
    total_rows = 3 * n + 6 * n
    if include_acceleration:
        total_rows += 6 * n
    if include_anchor:
        total_rows += 6 * n
    if include_posture:
        total_rows += 6 * n
    if include_margin:
        total_rows += 6 * n
    pattern = sparse.lil_matrix((total_rows, 6 * n), dtype=np.int8)
    row = 0
    for i in range(n):
        pattern[row : row + 3, 6 * i : 6 * (i + 1)] = 1
        row += 3
    for i in range(n):
        j = (i + 1) % n
        pattern[row : row + 6, 6 * i : 6 * (i + 1)] = 1
        pattern[row : row + 6, 6 * j : 6 * (j + 1)] = 1
        row += 6
    if include_acceleration:
        for i in range(n):
            for j in ((i - 1) % n, i, (i + 1) % n):
                pattern[row : row + 6, 6 * j : 6 * (j + 1)] = 1
            row += 6
    if include_anchor:
        for i in range(n):
            pattern[row : row + 6, 6 * i : 6 * (i + 1)] = 1
            row += 6
    if include_posture:
        for i in range(n):
            pattern[row : row + 6, 6 * i : 6 * (i + 1)] = 1
            row += 6
    if include_margin:
        for i in range(n):
            pattern[row : row + 6, 6 * i : 6 * (i + 1)] = 1
            row += 6
    return pattern.tocsr()


def trajectory_solution_frame(
    targets: pd.DataFrame | np.ndarray,
    beta_rad: np.ndarray,
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    compute_conditioning: bool = True,
) -> pd.DataFrame:
    frame = _trajectory_targets_frame(targets)
    beta = np.asarray(beta_rad, dtype=float).reshape(len(frame), 6)
    xyz = fk_from_beta_batch(
        beta,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
    )
    target_xyz = frame[TARGET_XYZ_COLS].to_numpy(dtype=float)
    theta = theta_from_beta_batch(beta, theta_sign=theta_sign)
    for j, col in enumerate(BETA_COLS):
        frame[col] = beta[:, j]
    for j, col in enumerate(THETA_COLS):
        frame[col] = theta[:, j]
    for j, col in enumerate(XYZ_COLS):
        frame[col] = xyz[:, j]
    frame["xyz_residual_mm"] = np.linalg.norm(xyz - target_xyz, axis=1) * 1000.0
    frame["canonical_posture_cost"] = [canonical_posture_penalty(row) for row in beta]
    if compute_conditioning:
        metrics = [
            jacobian_metrics(
                numerical_jacobian_beta(
                    row,
                    lengths_m=lengths_m,
                    p_end_local_m=p_end_local_m,
                    theta_sign=theta_sign,
                )
            )
            for row in beta
        ]
        for key in ("sigma1_m", "sigma2_m", "sigma3_m", "kappa"):
            frame[key] = [float(item[key]) for item in metrics]
    return frame


def evaluate_conditioning_policy(
    report: Mapping[str, Any],
    *,
    kappa_threshold: float,
    sigma3_min_m: float = 0.0015,
) -> dict[str, Any]:
    """Evaluate one candidate conditioning policy without registering it.

    The legacy strict gate continues to call this helper with a 150.0 kappa
    threshold.  Larger thresholds are evidence-sweep candidates only.
    """

    threshold = float(kappa_threshold)
    sigma_min = float(sigma3_min_m)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("conditioning kappa threshold must be finite and positive")
    if not np.isfinite(sigma_min) or sigma_min <= 0.0:
        raise ValueError("conditioning sigma3 minimum must be finite and positive")
    sigma_pass = bool(float(report.get("sigma3_p05_m", -np.inf)) >= sigma_min)
    kappa_pass = bool(float(report.get("kappa_p95", np.inf)) <= threshold)
    return {
        "sigma3_min_m": sigma_min,
        "kappa_threshold": threshold,
        "sigma3_gate_pass": sigma_pass,
        "kappa_gate_pass": kappa_pass,
        "conditioning_policy_gate_pass": bool(sigma_pass and kappa_pass),
    }


def evaluate_centerline_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    branch_gate = bool(
        float(report.get("residual_p95_mm", np.inf)) <= 2.0
        and float(report.get("residual_max_mm", np.inf)) <= 5.0
        and float(report.get("delta_beta_p95_deg", np.inf)) <= 2.0
        and float(report.get("seam_beta_rms_deg", np.inf)) <= 0.75
    )
    canonical_gate = bool(
        float(report.get("delta_beta_p95_deg", np.inf)) <= 1.0
        and float(report.get("delta_beta_max_deg", np.inf)) <= 2.0
        and float(report.get("delta2_beta_p95_deg", np.inf)) <= 0.25
        and float(report.get("seam_beta_rms_deg", np.inf)) <= 0.75
    )
    strict_conditioning = evaluate_conditioning_policy(
        report,
        kappa_threshold=150.0,
        sigma3_min_m=0.0015,
    )
    conditioning_gate = bool(strict_conditioning["conditioning_policy_gate_pass"])
    downstream_conditioning_gate = bool(strict_conditioning["sigma3_gate_pass"])
    return {
        "branch_gate_pass": branch_gate,
        "canonical_gate_pass": canonical_gate,
        "conditioning_gate_pass": conditioning_gate,
        "strict_conditioning_gate_pass": conditioning_gate,
        "downstream_admission_conditioning_gate_pass": downstream_conditioning_gate,
        "centerline_gate_pass": bool(branch_gate and canonical_gate and conditioning_gate),
        "downstream_admission_gate_pass": bool(
            branch_gate and canonical_gate and downstream_conditioning_gate
        ),
    }


def select_trajectory_stage(
    stage_outputs: Sequence[
        tuple[str, pd.DataFrame, Mapping[str, Any], np.ndarray]
    ],
    *,
    prefer_stage_acceptance: bool = True,
) -> tuple[str, pd.DataFrame, Mapping[str, Any], np.ndarray]:
    """Select the materialized stage accepted by the active job policy.

    ``stage_acceptance_gate_pass`` is deliberately distinct from the legacy
    ``centerline_gate_pass``.  Candidate conditioning policies can therefore
    keep the concrete optimizer stage that passed their threshold instead of
    falling through to a later, lower-residual but worse-conditioned stage.
    """

    if not stage_outputs:
        raise ValueError("trajectory stage selection requires at least one output")
    if prefer_stage_acceptance:
        accepted = [
            item
            for item in stage_outputs
            if bool(item[2].get("stage_acceptance_gate_pass", False))
        ]
        if accepted:
            return accepted[-1]
    centerline_pass = [
        item for item in stage_outputs if bool(item[2].get("centerline_gate_pass", False))
    ]
    if centerline_pass:
        return centerline_pass[-1]
    tracking_pass = [
        item for item in stage_outputs if bool(item[2].get("branch_gate_pass", False))
    ]
    if tracking_pass:
        return tracking_pass[-1]
    return min(
        stage_outputs,
        key=lambda item: (
            float(item[2].get("residual_p95_mm", np.inf)),
            float(item[2].get("residual_max_mm", np.inf)),
        ),
    )


def optimize_cyclic_trajectory(
    *,
    targets: pd.DataFrame | np.ndarray,
    initial_beta: np.ndarray,
    bounds: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    stages: Sequence[Mapping[str, float | str]] | None = None,
    max_nfev: int = 40,
    compute_conditioning: bool = True,
    stop_on_centerline_gate: bool = False,
    stage_decision: Callable[
        [pd.DataFrame, Mapping[str, Any]], Mapping[str, Any] | bool
    ]
    | None = None,
    select_on_stage_acceptance: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _trajectory_targets_frame(targets)
    target_xyz = frame[TARGET_XYZ_COLS].to_numpy(dtype=float)
    beta = np.asarray(initial_beta, dtype=float).reshape(len(frame), 6)
    limits = np.asarray(bounds, dtype=float).reshape(6, 2)
    anchor = beta.copy()
    stage_specs = list(DEFAULT_TRAJECTORY_STAGES if stages is None else stages)
    if not stage_specs:
        raise ValueError("at least one trajectory optimization stage is required")
    lower = np.tile(limits[:, 0], len(frame))
    upper = np.tile(limits[:, 1], len(frame))
    stage_outputs: list[tuple[str, pd.DataFrame, dict[str, Any], np.ndarray]] = []

    for stage_idx, stage in enumerate(stage_specs):
        name = str(stage.get("name", f"stage_{stage_idx}"))
        lambda_velocity = float(stage.get("lambda_velocity", 1.0))
        lambda_acceleration = float(stage.get("lambda_acceleration", 0.0))
        lambda_anchor = float(stage.get("lambda_anchor", 0.0))
        lambda_posture = float(stage.get("lambda_posture", 0.0))
        lambda_margin = float(stage.get("lambda_margin", 0.0))
        soft_margin_deg = float(stage.get("soft_margin_deg", 0.25))
        jac_pattern = cyclic_trajectory_jac_sparsity(
            n_points=len(frame),
            include_acceleration=lambda_acceleration > 0.0,
            include_anchor=lambda_anchor > 0.0,
            include_posture=lambda_posture > 0.0,
            include_margin=lambda_margin > 0.0,
        )

        def objective(flat: np.ndarray) -> np.ndarray:
            return cyclic_trajectory_residual(
                flat,
                targets_xyz=target_xyz,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
                anchor_beta=anchor,
                bounds=limits,
                lambda_velocity=lambda_velocity,
                lambda_acceleration=lambda_acceleration,
                lambda_anchor=lambda_anchor,
                lambda_posture=lambda_posture,
                lambda_margin=lambda_margin,
                soft_margin_deg=soft_margin_deg,
            )

        result = least_squares(
            objective,
            np.clip(beta.reshape(-1), lower, upper),
            bounds=(lower, upper),
            method="trf",
            jac_sparsity=jac_pattern,
            x_scale="jac",
            max_nfev=int(max_nfev),
            xtol=1.0e-8,
            ftol=1.0e-8,
            gtol=1.0e-8,
        )
        beta = np.asarray(result.x, dtype=float).reshape(len(frame), 6)
        solution_frame = trajectory_solution_frame(
            frame,
            beta,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
            compute_conditioning=compute_conditioning,
        )
        stage_report = smoothness_report(solution_frame)
        stage_report.update(
            {
                "stage": name,
                "optimizer_success": bool(result.success),
                "optimizer_status": int(result.status),
                "optimizer_nfev": int(result.nfev),
                "optimizer_cost": float(result.cost),
                "lambda_velocity": lambda_velocity,
                "lambda_acceleration": lambda_acceleration,
                "lambda_anchor": lambda_anchor,
                "lambda_posture": lambda_posture,
                "lambda_margin": lambda_margin,
                "soft_margin_deg": soft_margin_deg,
            }
        )
        stage_report.update(evaluate_centerline_gates(stage_report))
        if stage_decision is None:
            stage_accepted = bool(stage_report["centerline_gate_pass"])
        else:
            raw_decision = stage_decision(solution_frame, dict(stage_report))
            if isinstance(raw_decision, Mapping):
                stage_report.update(dict(raw_decision))
                stage_accepted = bool(
                    raw_decision.get(
                        "stage_acceptance_gate_pass",
                        raw_decision.get("job_gate_pass", False),
                    )
                )
            else:
                stage_accepted = bool(raw_decision)
        stage_report["stage_acceptance_gate_pass"] = stage_accepted
        stage_outputs.append((name, solution_frame, stage_report, beta.copy()))
        stop_gate = (
            stage_accepted
            if bool(select_on_stage_acceptance)
            else bool(stage_report["centerline_gate_pass"])
        )
        if bool(stop_on_centerline_gate) and stop_gate:
            break

    selected_name, selected_frame, selected_report, _selected_beta = (
        select_trajectory_stage(
            stage_outputs,
            prefer_stage_acceptance=bool(select_on_stage_acceptance),
        )
    )
    report = dict(selected_report)
    report["selected_stage"] = selected_name
    report["stage_history"] = [dict(item[2]) for item in stage_outputs]
    report["input_branch_hash"] = hashlib.sha256(np.round(anchor, 10).tobytes()).hexdigest()
    report["output_branch_hash"] = branch_hash(selected_frame)
    return selected_frame, report


def local_beta_consistency_report(
    df: pd.DataFrame,
    *,
    radius_mm: float = 10.0,
    max_neighbors: int | None = None,
    multi_branch_threshold_deg: float = 3.0,
) -> dict[str, Any]:
    if df.empty:
        return {
            "tube10_beta_rms_p95_deg": float("inf"),
            "multi_branch_ratio": 1.0,
            "neighbor_pair_count": 0,
        }
    xyz = df[XYZ_COLS].to_numpy(dtype=float)
    beta = df[BETA_COLS].to_numpy(dtype=float)
    radius_m = float(radius_mm) / 1000.0
    if max_neighbors is None:
        nn = NearestNeighbors().fit(xyz)
        distance_rows, index_rows = nn.radius_neighbors(xyz, radius=radius_m, return_distance=True)
    else:
        k = min(max(int(max_neighbors), 2), len(df))
        nn = NearestNeighbors(n_neighbors=k).fit(xyz)
        distance, index = nn.kneighbors(xyz)
        distance_rows = [row[(row > 1.0e-12) & (row <= radius_m)] for row in distance]
        index_rows = [
            index[i][(distance[i] > 1.0e-12) & (distance[i] <= radius_m)]
            for i in range(len(df))
        ]
    pair_values: list[float] = []
    point_max = np.zeros(len(df), dtype=float)
    point_has_neighbor = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        distances_i = np.asarray(distance_rows[i], dtype=float)
        indices_i = np.asarray(index_rows[i], dtype=np.int64)
        nonself = distances_i > 1.0e-12
        indices_i = indices_i[nonself]
        if len(indices_i) == 0:
            continue
        values = np.sqrt(np.mean(np.square(beta[indices_i] - beta[i]), axis=1)) * 180.0 / math.pi
        pair_values.extend(values.tolist())
        point_max[i] = float(np.max(values))
        point_has_neighbor[i] = True
    if not pair_values:
        return {
            "tube10_beta_rms_p95_deg": float("inf"),
            "multi_branch_ratio": 1.0,
            "neighbor_pair_count": 0,
            "neighbor_coverage_ratio": 0.0,
        }
    return {
        "tube10_beta_rms_p95_deg": float(np.percentile(np.asarray(pair_values), 95)),
        "multi_branch_ratio": float(np.mean(point_max[point_has_neighbor] > float(multi_branch_threshold_deg))),
        "neighbor_pair_count": int(len(pair_values)),
        "neighbor_coverage_ratio": float(np.mean(point_has_neighbor)),
    }


def make_tube_frames(centerline_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(centerline_xyz, dtype=float).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
    tau = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    norm = np.linalg.norm(tau, axis=1, keepdims=True)
    tau = tau / np.maximum(norm, 1.0e-12)
    n1 = np.empty_like(tau)
    ref0 = np.asarray([0.0, 0.0, 1.0])
    for i, t in enumerate(tau):
        ref = ref0 if abs(float(np.dot(t, ref0))) < 0.9 else np.asarray([0.0, 1.0, 0.0])
        v = ref - np.dot(ref, t) * t
        if i > 0 and np.dot(v, n1[i - 1]) < 0.0:
            v = -v
        n1[i] = v / max(float(np.linalg.norm(v)), 1.0e-12)
    n2 = np.cross(tau, n1)
    n2 = n2 / np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1.0e-12)
    return tau, n1, n2


def make_normal_tube_targets(centerline: pd.DataFrame, *, offsets_mm: Iterable[float]) -> pd.DataFrame:
    offsets = [float(v) for v in offsets_mm]
    xyz = centerline[TARGET_XYZ_COLS].to_numpy(dtype=float)
    _tau, n1, n2 = make_tube_frames(xyz)
    rows: list[dict[str, Any]] = []
    for i, row in centerline.reset_index(drop=True).iterrows():
        base = row[TARGET_XYZ_COLS].to_numpy(dtype=float)
        for dn1 in offsets:
            for dn2 in offsets:
                target = base + (dn1 / 1000.0) * n1[i] + (dn2 / 1000.0) * n2[i]
                rec = row.to_dict()
                rec.update(
                    {
                        "tube_idx": len(rows),
                        "delta_n1_mm": float(dn1),
                        "delta_n2_mm": float(dn2),
                        "tube_offset_id": f"n1_{dn1:g}_n2_{dn2:g}",
                        "is_centerline": bool(abs(dn1) < 1.0e-12 and abs(dn2) < 1.0e-12),
                        "x_target_m": float(target[0]),
                        "y_target_m": float(target[1]),
                        "z_target_m": float(target[2]),
                    }
                )
                rows.append(rec)
    return pd.DataFrame(rows)


def weighted_damped_pinv(jac: np.ndarray, *, damping: float = 1.0e-3, weights: np.ndarray | None = None) -> np.ndarray:
    j = np.asarray(jac, dtype=float).reshape(3, 6)
    w = np.asarray(weights if weights is not None else [4, 4, 2, 2, 1, 1], dtype=float).reshape(6)
    winv = np.diag(1.0 / np.maximum(w, 1.0e-12))
    return winv @ j.T @ np.linalg.inv(j @ winv @ j.T + float(damping) ** 2 * np.eye(3))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def write_markdown_table(df: pd.DataFrame, path: Path, *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if df.empty:
        lines.append("_empty_")
    else:
        shown = df.copy()
        headers = [str(c) for c in shown.columns]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in shown.itertuples(index=False):
            vals = []
            for v in row:
                if isinstance(v, float):
                    vals.append(f"{v:.6g}" if math.isfinite(v) else "")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)
