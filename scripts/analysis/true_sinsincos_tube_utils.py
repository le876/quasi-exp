#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors


XYZ_COLS = ["x_m", "y_m", "z_m"]
TARGET_XYZ_COLS = ["x_target_m", "y_target_m", "z_target_m"]
BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
U_DEG_COLS = ["u_a_deg", "u_b_deg", "u_eta"]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def theta_from_beta_batch(beta: np.ndarray, *, theta_sign: float = -1.0) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).reshape(-1, 6)
    theta = np.empty((len(beta), 30), dtype=float)
    for section, (odd_col, even_col) in enumerate(((0, 1), (2, 3), (4, 5))):
        start = section * 10
        theta[:, start : start + 10 : 2] = beta[:, odd_col : odd_col + 1]
        theta[:, start + 1 : start + 10 : 2] = beta[:, even_col : even_col + 1]
    return theta * float(theta_sign)


def beta_from_u_deg(
    u: np.ndarray,
    *,
    s10: float,
    s20: float,
    ds1: float,
    ds2: float,
    dq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float).reshape(-1, 3)
    a = np.deg2rad(u[:, 0])
    b = np.deg2rad(u[:, 1])
    eta = u[:, 2]
    s1 = float(s10) + float(ds1) * eta
    s2 = float(s20) + float(ds2) * eta
    q = 1.0 + float(dq) * eta
    beta = np.column_stack([s1 * a, s1 * b, s2 * a, s2 * b, q * a, q * b])
    return beta, s1, s2, q


def infer_u_mapping_params(pool: pd.DataFrame) -> dict[str, float]:
    """Infer the affine layer-field mapping used by A1-like pools."""
    def clean(value: float) -> float:
        if abs(float(value)) < 1.0e-12:
            return 0.0
        return float(round(float(value), 12))

    if not {"u_eta", "s1", "s2"}.issubset(pool.columns):
        return {"s10": 0.125, "s20": 0.25, "ds1": 0.0, "ds2": 0.0, "dq": 0.0}
    eta = pool["u_eta"].to_numpy(dtype=float)

    def fit_line(col: str) -> tuple[float, float]:
        y = pool[col].to_numpy(dtype=float)
        finite = np.isfinite(eta) & np.isfinite(y)
        if finite.sum() < 2 or np.nanstd(eta[finite]) < 1.0e-12:
            return float(np.nanmedian(y[finite])) if finite.any() else 0.0, 0.0
        slope, intercept = np.polyfit(eta[finite], y[finite], 1)
        return float(intercept), float(slope)

    s10, ds1 = fit_line("s1")
    s20, ds2 = fit_line("s2")
    if "q_distal" in pool.columns:
        q0, dq = fit_line("q_distal")
        if math.isfinite(q0) and abs(q0) > 1.0e-9:
            dq = dq / q0
    else:
        dq = 0.0
    return {"s10": clean(s10), "s20": clean(s20), "ds1": clean(ds1), "ds2": clean(ds2), "dq": clean(float(dq))}


def make_true_sinsincos_ellipse(
    *,
    ellipse_id: str,
    center: tuple[float, float, float],
    amp_xy_mm: float,
    n_points: int = 360,
    phase_rad: float = 0.0,
    include_endpoint: bool = False,
) -> pd.DataFrame:
    count = int(n_points) + (1 if include_endpoint else 0)
    t = np.linspace(0.0, 2.0 * math.pi, count, endpoint=include_endpoint, dtype=float)
    amp_xy_m = float(amp_xy_mm) / 1000.0
    amp_z_m = 1.5 * amp_xy_m
    cx, cy, cz = (float(v) for v in center)
    x = cx + amp_xy_m * np.sin(t + float(phase_rad))
    y = cy + amp_xy_m * np.sin(t + float(phase_rad))
    z = cz + amp_z_m * np.cos(t + float(phase_rad))
    return pd.DataFrame(
        {
            "ellipse_id": str(ellipse_id),
            "angle_idx": np.arange(count, dtype=np.int64),
            "angle_rad": t,
            "angle_deg": t * 180.0 / math.pi,
            "x_target_m": x,
            "y_target_m": y,
            "z_target_m": z,
            "center_x": cx,
            "center_y": cy,
            "center_z": cz,
            "amp_xy_mm": float(amp_xy_mm),
            "amp_z_mm": float(1.5 * float(amp_xy_mm)),
            "trajectory_family": "sin_sin_cos",
        }
    )


def trajectory_geometry_metrics(xyz: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if len(pts) < 4:
        raise ValueError("trajectory must contain at least four points")
    centered = pts - pts.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    s1 = max(float(singular[0]), 1.0e-15)
    s2 = float(singular[1]) if len(singular) > 1 else 0.0
    s3 = float(singular[2]) if len(singular) > 2 else 0.0
    closed_loop_gap = float(np.linalg.norm(pts[0] - pts[-1]))
    ranges = np.ptp(pts, axis=0)
    amp_x = float(ranges[0] / 2.0)
    amp_y = float(ranges[1] / 2.0)
    amp_z = float(ranges[2] / 2.0)
    amp_xy_mean = max((amp_x + amp_y) / 2.0, 1.0e-15)
    # Area in the principal 2D plane catches collinear same-phase trajectories.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    uv = centered @ vt[:2].T
    if np.linalg.norm(pts[0] - pts[-1]) > 1.0e-8:
        uv_area = np.vstack([uv, uv[0]])
    else:
        uv_area = uv
    area = 0.5 * abs(
        float(np.dot(uv_area[:-1, 0], uv_area[1:, 1]) - np.dot(uv_area[1:, 0], uv_area[:-1, 1]))
    )
    rank2 = bool((s2 / s1) > 0.3 and (s3 / s1) < 1.0e-3 and area > 1.0e-10)
    return {
        "svd1_mm": float(singular[0] * 1000.0),
        "svd2_mm": float(s2 * 1000.0),
        "svd3_mm": float(s3 * 1000.0),
        "svd2_over_svd1": float(s2 / s1),
        "svd3_over_svd1": float(s3 / s1),
        "closed_loop_gap_m": closed_loop_gap,
        "amp_x_mm": amp_x * 1000.0,
        "amp_y_mm": amp_y * 1000.0,
        "amp_z_mm": amp_z * 1000.0,
        "amp_z_over_xy_mean": float(amp_z / amp_xy_mean),
        "polygon_area_m2": area,
        "rank2_gate_pass": rank2,
    }


def recompute_support_for_targets(
    targets: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    pool_label: str,
    tube_radius_mm: float = 15.0,
    beta_pair_radius_mm: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_target = [c for c in TARGET_XYZ_COLS if c not in targets.columns]
    missing_pool = [c for c in XYZ_COLS if c not in pool.columns]
    if missing_target or missing_pool:
        raise ValueError(f"missing target={missing_target} pool={missing_pool}")
    target_xyz = targets[TARGET_XYZ_COLS].to_numpy(dtype=float)
    pool_xyz = pool[XYZ_COLS].to_numpy(dtype=float)
    nn1 = NearestNeighbors(n_neighbors=1).fit(pool_xyz)
    dist, idx = nn1.kneighbors(target_xyz, n_neighbors=1)
    nearest = pool_xyz[idx[:, 0]]
    radius_nn = NearestNeighbors().fit(pool_xyz)
    tube_indices = radius_nn.radius_neighbors(target_xyz, radius=float(tube_radius_mm) / 1000.0, return_distance=False)
    counts = np.asarray([len(v) for v in tube_indices], dtype=float)
    dist_mm = dist[:, 0] * 1000.0
    diff_mm = (nearest - target_xyz) * 1000.0

    beta_p95 = float("nan")
    has_beta = all(c in pool.columns for c in BETA_COLS)
    local_unique = np.unique(np.concatenate([v for v in tube_indices if len(v) > 0])) if np.any(counts > 0) else np.empty((0,), dtype=np.int64)
    if has_beta and len(local_unique) >= 2:
        local_xyz = pool_xyz[local_unique]
        local_beta = pool.iloc[local_unique][BETA_COLS].to_numpy(dtype=float)
        k = min(16, len(local_unique))
        nn_local = NearestNeighbors(n_neighbors=k).fit(local_xyz)
        local_dist, local_idx = nn_local.kneighbors(local_xyz)
        vals: list[float] = []
        pair_radius_m = float(beta_pair_radius_mm) / 1000.0
        for i in range(len(local_xyz)):
            mask = (local_dist[i] > 1.0e-12) & (local_dist[i] <= pair_radius_m)
            if np.any(mask):
                diff = local_beta[local_idx[i][mask]] - local_beta[i]
                vals.extend((np.sqrt(np.mean(np.square(diff), axis=1)) * 180.0 / math.pi).tolist())
        if vals:
            beta_p95 = float(np.percentile(vals, 95))

    group_cols = [c for c in ["ellipse_id", "candidate_id", "amp_xy_mm", "amp_z_mm", "center_x", "center_y", "center_z"] if c in targets.columns]
    tmp = targets[group_cols].copy() if group_cols else pd.DataFrame(index=targets.index)
    tmp["true_nn_mm"] = dist_mm
    tmp["true_tube_count_15mm"] = counts
    summaries: list[dict[str, Any]] = []
    if not group_cols:
        groups = [((), tmp)]
    elif len(group_cols) == 1:
        groups = tmp.groupby(group_cols[0], dropna=False, sort=False)
    else:
        groups = tmp.groupby(group_cols, dropna=False, sort=False)
    for key, part in groups:
        row: dict[str, Any] = {"pool_label": str(pool_label)}
        if group_cols:
            key_tuple = key if isinstance(key, tuple) else (key,)
            row.update({col: val for col, val in zip(group_cols, key_tuple)})
        vals = part["true_nn_mm"].to_numpy(dtype=float)
        cts = part["true_tube_count_15mm"].to_numpy(dtype=float)
        row.update(
            {
                "true_nn_mean_mm": float(np.mean(vals)),
                "true_nn_p50_mm": float(np.percentile(vals, 50)),
                "true_nn_p95_mm": float(np.percentile(vals, 95)),
                "true_nn_max_mm": float(np.max(vals)),
                "true_tube_count_min": float(np.min(cts)),
                "true_tube_count_p10": float(np.percentile(cts, 10)),
                "true_tube_count_p50": float(np.percentile(cts, 50)),
                "true_tube_count_p95": float(np.percentile(cts, 95)),
                "true_tube_beta_rms_p95_deg": beta_p95,
                "true_support_gate_pass": bool(np.percentile(vals, 95) <= 8.0 and np.percentile(cts, 10) >= 16.0),
                "true_branch_gate_pass": bool(np.isfinite(beta_p95) and beta_p95 <= 1.0 and np.percentile(cts, 10) >= 16.0),
            }
        )
        summaries.append(row)

    per_angle = targets.copy().reset_index(drop=True)
    per_angle["pool_label"] = str(pool_label)
    per_angle["true_nn_mm"] = dist_mm
    per_angle["true_tube_count_15mm"] = counts
    per_angle["nearest_x_diff_mm"] = diff_mm[:, 0]
    per_angle["nearest_y_diff_mm"] = diff_mm[:, 1]
    per_angle["nearest_z_diff_mm"] = diff_mm[:, 2]
    if has_beta:
        nearest_beta = pool.iloc[idx[:, 0]][BETA_COLS].to_numpy(dtype=float)
        if all(c in targets.columns for c in BETA_COLS):
            target_beta = targets[BETA_COLS].to_numpy(dtype=float)
            per_angle["nearest_beta_dist_to_A1_deg"] = np.sqrt(np.mean(np.square(nearest_beta - target_beta), axis=1)) * 180.0 / math.pi
        else:
            per_angle["nearest_beta_dist_to_A1_deg"] = np.nan
    return pd.DataFrame(summaries), per_angle


def make_tube_frames(centerline_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(centerline_xyz, dtype=float).reshape(-1, 3)
    closed = np.linalg.norm(pts[0] - pts[-1]) < 1.0e-8
    core = pts[:-1] if closed else pts
    n = len(core)
    tau = np.empty((n, 3), dtype=float)
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        v = core[next_i] - core[prev_i]
        norm = np.linalg.norm(v)
        tau[i] = v / max(norm, 1.0e-12)
    n1 = np.empty_like(tau)
    n2 = np.empty_like(tau)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, tau[0]))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    n1[0] = ref - np.dot(ref, tau[0]) * tau[0]
    n1[0] /= max(np.linalg.norm(n1[0]), 1.0e-12)
    n2[0] = np.cross(tau[0], n1[0])
    n2[0] /= max(np.linalg.norm(n2[0]), 1.0e-12)
    for i in range(1, n):
        v = n1[i - 1] - np.dot(n1[i - 1], tau[i]) * tau[i]
        if np.linalg.norm(v) < 1.0e-9:
            v = ref - np.dot(ref, tau[i]) * tau[i]
        n1[i] = v / max(np.linalg.norm(v), 1.0e-12)
        if np.dot(n1[i], n1[i - 1]) < 0:
            n1[i] *= -1.0
        n2[i] = np.cross(tau[i], n1[i])
        n2[i] /= max(np.linalg.norm(n2[i]), 1.0e-12)
    if closed:
        tau = np.vstack([tau, tau[0]])
        n1 = np.vstack([n1, n1[0]])
        n2 = np.vstack([n2, n2[0]])
    return tau, n1, n2


def make_tube_targets(centerline: pd.DataFrame, offsets_mm: Iterable[float]) -> pd.DataFrame:
    offsets = [float(v) for v in offsets_mm]
    xyz = centerline[TARGET_XYZ_COLS].to_numpy(dtype=float)
    tau, n1, n2 = make_tube_frames(xyz)
    rows: list[dict[str, Any]] = []
    for row_i, row in centerline.reset_index(drop=True).iterrows():
        p = xyz[row_i]
        for dt in offsets:
            for dn1 in offsets:
                for dn2 in offsets:
                    target = p + (dt / 1000.0) * tau[row_i] + (dn1 / 1000.0) * n1[row_i] + (dn2 / 1000.0) * n2[row_i]
                    rec = row.to_dict()
                    rec.update(
                        {
                            "tube_idx": len(rows),
                            "x_target_m": target[0],
                            "y_target_m": target[1],
                            "z_target_m": target[2],
                            "delta_t_mm": dt,
                            "delta_n1_mm": dn1,
                            "delta_n2_mm": dn2,
                            "tau_x": tau[row_i, 0],
                            "tau_y": tau[row_i, 1],
                            "tau_z": tau[row_i, 2],
                            "n1_x": n1[row_i, 0],
                            "n1_y": n1[row_i, 1],
                            "n1_z": n1[row_i, 2],
                            "n2_x": n2[row_i, 0],
                            "n2_y": n2[row_i, 1],
                            "n2_z": n2[row_i, 2],
                            "is_centerline": bool(abs(dt) < 1.0e-12 and abs(dn1) < 1.0e-12 and abs(dn2) < 1.0e-12),
                        }
                    )
                    rows.append(rec)
    return pd.DataFrame(rows)


def attach_fk_from_u(
    u_df: pd.DataFrame,
    *,
    fk_func: Any,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    mapping: dict[str, float],
) -> pd.DataFrame:
    u = u_df[U_DEG_COLS].to_numpy(dtype=float)
    beta, s1, s2, q = beta_from_u_deg(u, **mapping)
    theta = theta_from_beta_batch(beta, theta_sign=theta_sign)
    xyz = fk_func(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    out = u_df.copy().reset_index(drop=True)
    out["u_a_rad"] = np.deg2rad(out["u_a_deg"].to_numpy(dtype=float))
    out["u_b_rad"] = np.deg2rad(out["u_b_deg"].to_numpy(dtype=float))
    out["s1"] = s1
    out["s2"] = s2
    out["q_distal"] = q
    for i, col in enumerate(BETA_COLS):
        out[col] = beta[:, i]
    for i, col in enumerate(THETA_COLS):
        out[col] = theta[:, i]
    for i, col in enumerate(XYZ_COLS):
        out[col] = xyz[:, i]
    return out


def optimize_u_for_target(
    target_xyz: np.ndarray,
    init_u_deg: np.ndarray,
    *,
    fk_func: Any,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    mapping: dict[str, float],
    center_u_deg: np.ndarray | None = None,
    max_nfev: int = 100,
    sigma_x_mm: float = 2.0,
    lambda_u: float = 0.05,
) -> tuple[np.ndarray, float, bool, int]:
    target = np.asarray(target_xyz, dtype=float).reshape(3)
    init = np.asarray(init_u_deg, dtype=float).reshape(3)
    center = init if center_u_deg is None else np.asarray(center_u_deg, dtype=float).reshape(3)

    def residual(u: np.ndarray) -> np.ndarray:
        beta, _s1, _s2, _q = beta_from_u_deg(u.reshape(1, 3), **mapping)
        theta = theta_from_beta_batch(beta, theta_sign=theta_sign)
        xyz = fk_func(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)[0]
        fk_res = (xyz - target) * 1000.0 / max(float(sigma_x_mm), 1.0e-9)
        u_res = math.sqrt(max(float(lambda_u), 0.0)) * (u - center)
        return np.concatenate([fk_res, u_res])

    res = least_squares(
        residual,
        init,
        bounds=(np.array([-15.0, -15.0, -1.0]), np.array([15.0, 15.0, 1.0])),
        max_nfev=int(max_nfev),
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    beta, _s1, _s2, _q = beta_from_u_deg(res.x.reshape(1, 3), **mapping)
    theta = theta_from_beta_batch(beta, theta_sign=theta_sign)
    xyz = fk_func(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)[0]
    err_mm = float(np.linalg.norm(xyz - target) * 1000.0)
    return res.x.astype(float), err_mm, bool(res.success), int(res.nfev)


def write_markdown_table(df: pd.DataFrame, path: Path, *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if df.empty:
        lines.append("_empty_")
    else:
        cols = [str(c) for c in df.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in df.itertuples(index=False):
            vals = []
            for value in row:
                if isinstance(value, float):
                    vals.append(f"{value:.6g}" if math.isfinite(value) else "")
                else:
                    vals.append(str(value))
            lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
