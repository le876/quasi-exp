#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402


XYZ_COLS = ["x_m", "y_m", "z_m"]
BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
U_COLS = ["u_a_deg", "u_b_deg", "u_eta"]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _parse_float_list(raw: str) -> list[float]:
    return [float(v.strip()) for v in str(raw).split(",") if v.strip()]


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def theta_from_beta_batch(beta: np.ndarray, *, theta_sign: float) -> np.ndarray:
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


def attach_fk(
    u_df: pd.DataFrame,
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    s10: float,
    s20: float,
    ds1: float,
    ds2: float,
    dq: float,
) -> pd.DataFrame:
    u = u_df[U_COLS].to_numpy(dtype=float)
    beta, s1, s2, q = beta_from_u_deg(u, s10=s10, s20=s20, ds1=ds1, ds2=ds2, dq=dq)
    theta = theta_from_beta_batch(beta, theta_sign=theta_sign)
    xyz = fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    out = u_df.copy().reset_index(drop=True)
    out["u_a_rad"] = np.deg2rad(out["u_a_deg"].to_numpy(dtype=float))
    out["u_b_rad"] = np.deg2rad(out["u_b_deg"].to_numpy(dtype=float))
    out["rho_deg"] = np.sqrt(np.square(out["u_a_deg"].to_numpy(dtype=float)) + np.square(out["u_b_deg"].to_numpy(dtype=float)))
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


def attach_jacobian(
    df: pd.DataFrame,
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    s10: float,
    s20: float,
    ds1: float,
    ds2: float,
    dq: float,
    delta: float,
    sigma3_min_m: float,
    kappa_max: float,
    batch_size: int,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    u = out[U_COLS].to_numpy(dtype=float)
    singular = np.empty((len(out), 3), dtype=float)
    for start in range(0, len(out), int(batch_size)):
        end = min(len(out), start + int(batch_size))
        base = u[start:end]
        jac = np.empty((end - start, 3, 3), dtype=float)
        for j, scale in enumerate((1.0, 1.0, 1.0)):
            step = np.zeros_like(base)
            step[:, j] = float(delta) * scale
            plus = np.clip(base + step, [-15.0, -15.0, -1.0], [15.0, 15.0, 1.0])
            minus = np.clip(base - step, [-15.0, -15.0, -1.0], [15.0, 15.0, 1.0])
            beta_p, _s1, _s2, _q = beta_from_u_deg(plus, s10=s10, s20=s20, ds1=ds1, ds2=ds2, dq=dq)
            beta_m, _s1, _s2, _q = beta_from_u_deg(minus, s10=s10, s20=s20, ds1=ds1, ds2=ds2, dq=dq)
            xyz_p = fk_dh_batch(theta_from_beta_batch(beta_p, theta_sign=theta_sign), lengths_m=lengths_m, p_end_local_m=p_end_local_m)
            xyz_m = fk_dh_batch(theta_from_beta_batch(beta_m, theta_sign=theta_sign), lengths_m=lengths_m, p_end_local_m=p_end_local_m)
            denom = (plus[:, j] - minus[:, j]).reshape(-1, 1)
            denom = np.where(np.abs(denom) < 1.0e-12, 2.0 * float(delta), denom)
            jac[:, :, j] = (xyz_p - xyz_m) / denom
        singular[start:end, :] = np.linalg.svd(jac, compute_uv=False)
    out["sigma1_m"] = singular[:, 0]
    out["sigma2_m"] = singular[:, 1]
    out["sigma3_m"] = singular[:, 2]
    out["kappa"] = singular[:, 0] / np.maximum(singular[:, 2], 1.0e-12)
    out["jacobian_gate_pass"] = (out["sigma3_m"] >= float(sigma3_min_m)) & (out["kappa"] <= float(kappa_max))
    return out


def ellipse_points(row: pd.Series, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False, dtype=float)
    xyz = np.empty((len(t), 3), dtype=float)
    xyz[:, 0] = float(row["center_x"]) + float(row["amp_xy_mm"]) / 1000.0 * np.sin(t + float(row["phase_x"]))
    xyz[:, 1] = float(row["center_y"]) + float(row["amp_xy_mm"]) / 1000.0 * np.sin(t + float(row["phase_y"]))
    xyz[:, 2] = float(row["center_z"]) + float(row["amp_z_mm"]) / 1000.0 * np.sin(t + float(row["phase_z"]))
    return xyz, t


def select_candidates(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.candidate_csv)
    frames = []
    for amp, limit in [(float(args.target_amp_xy_mm), int(args.max_target_candidates))]:
        sub = df[np.isclose(df["amp_xy_mm"].to_numpy(dtype=float), amp)].copy()
        if sub.empty:
            raise SystemExit(f"no candidates for target amp {amp}")
        sub["_x_abs"] = np.abs(sub["nearest_x_mean_diff_mm"].to_numpy(dtype=float))
        sub = sub.sort_values(["nn_p95_mm", "_x_abs", "nn_max_mm"], ascending=[True, True, True]).head(limit)
        sub["selection_role"] = "target"
        frames.append(sub)
    for amp in _parse_float_list(args.control_amp_xy_mm):
        sub = df[np.isclose(df["amp_xy_mm"].to_numpy(dtype=float), amp)].copy()
        if sub.empty:
            continue
        if "support_gate_pass" in sub.columns:
            sub["_support"] = _as_bool(sub["support_gate_pass"])
        else:
            sub["_support"] = False
        sub["_x_abs"] = np.abs(sub["nearest_x_mean_diff_mm"].to_numpy(dtype=float))
        sub = sub.sort_values(["_support", "nn_p95_mm", "_x_abs"], ascending=[False, True, True]).head(int(args.max_control_candidates))
        sub["selection_role"] = "control"
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    return out.drop(columns=[c for c in ["_x_abs", "_support"] if c in out.columns])


def support_metrics_for_pool(
    pool: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    n_points: int,
    tube_radius_mm: float,
    beta_pair_radius_mm: float,
    min_tube_count_p10: int,
    beta_gate_deg: float,
    max_nn_p95_mm: float,
    max_x_bias_mm: float,
    pool_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    xyz = pool[XYZ_COLS].to_numpy(dtype=float)
    beta = pool[BETA_COLS].to_numpy(dtype=float)
    nn1 = NearestNeighbors(n_neighbors=1).fit(xyz)
    radius_nn = NearestNeighbors().fit(xyz)
    summary_rows: list[dict[str, Any]] = []
    point_rows: list[pd.DataFrame] = []
    tube_radius_m = float(tube_radius_mm) / 1000.0
    pair_radius_m = float(beta_pair_radius_mm) / 1000.0
    for _idx, cand in candidates.iterrows():
        target, angle = ellipse_points(cand, n_points)
        dist, idx = nn1.kneighbors(target, n_neighbors=1)
        nearest = xyz[idx[:, 0]]
        rad_idx = radius_nn.radius_neighbors(target, radius=tube_radius_m, return_distance=False)
        counts = np.asarray([len(v) for v in rad_idx], dtype=float)
        x_diff_mm = (nearest[:, 0] - target[:, 0]) * 1000.0
        dist_mm = dist[:, 0] * 1000.0
        local_unique = np.unique(np.concatenate([v for v in rad_idx if len(v) > 0])) if np.any(counts > 0) else np.empty((0,), dtype=np.int64)
        thickness2 = float("nan")
        thickness3 = float("nan")
        if len(local_unique) >= 6:
            cov = np.cov((xyz[local_unique] - xyz[local_unique].mean(axis=0)).T)
            vals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))[::-1]
            thickness2 = float(math.sqrt(vals[1]) * 1000.0)
            thickness3 = float(math.sqrt(vals[2]) * 1000.0)
        beta_p95 = float("nan")
        if len(local_unique) >= 2:
            local_xyz = xyz[local_unique]
            local_beta = beta[local_unique]
            k = min(16, len(local_unique))
            nn_local = NearestNeighbors(n_neighbors=k).fit(local_xyz)
            local_dist, local_idx = nn_local.kneighbors(local_xyz)
            vals: list[float] = []
            for i in range(len(local_xyz)):
                mask = (local_dist[i] > 1.0e-12) & (local_dist[i] <= pair_radius_m)
                if not np.any(mask):
                    continue
                diff = local_beta[local_idx[i][mask]] - local_beta[i]
                vals.extend((np.sqrt(np.mean(np.square(diff), axis=1)) * 180.0 / math.pi).tolist())
            if vals:
                beta_p95 = float(np.percentile(vals, 95))
        row = cand.to_dict()
        row.update(
            {
                "pool_label": pool_label,
                "nn_mean_mm": float(np.mean(dist_mm)),
                "nn_p95_mm": float(np.percentile(dist_mm, 95)),
                "nn_max_mm": float(np.max(dist_mm)),
                "nearest_x_mean_diff_mm": float(np.mean(x_diff_mm)),
                "nearest_x_p95_abs_diff_mm": float(np.percentile(np.abs(x_diff_mm), 95)),
                "tube_count_p10": float(np.percentile(counts, 10)),
                "tube_count_median": float(np.percentile(counts, 50)),
                "tube_count_min": float(np.min(counts)) if len(counts) else 0.0,
                "tube_unique_rows": int(len(local_unique)),
                "tube_normal_thickness_2_mm": thickness2,
                "tube_normal_thickness_3_mm": thickness3,
                "tube_beta_rms_p95_deg": beta_p95,
            }
        )
        row["support_gate_pass"] = bool(row["nn_p95_mm"] <= float(max_nn_p95_mm) and abs(row["nearest_x_mean_diff_mm"]) <= float(max_x_bias_mm))
        row["branch_gate_pass"] = bool(
            row["support_gate_pass"]
            and np.isfinite(beta_p95)
            and beta_p95 <= float(beta_gate_deg)
            and row["tube_count_p10"] >= float(min_tube_count_p10)
            and np.isfinite(thickness2)
            and np.isfinite(thickness3)
            and thickness2 >= 5.0
            and thickness3 >= 3.0
        )
        summary_rows.append(row)
        point_rows.append(
            pd.DataFrame(
                {
                    "pool_label": pool_label,
                    "candidate_id": cand["candidate_id"],
                    "selection_role": cand.get("selection_role", ""),
                    "point_index": np.arange(len(target), dtype=np.int64),
                    "angle_rad": angle,
                    "target_x_m": target[:, 0],
                    "target_y_m": target[:, 1],
                    "target_z_m": target[:, 2],
                    "nearest_dist_mm": dist_mm,
                    "nearest_x_diff_mm": x_diff_mm,
                    "tube_count_15mm": counts,
                    "thin_by_dist": dist_mm > float(max_nn_p95_mm),
                    "thin_by_x": np.abs(x_diff_mm) > float(max_x_bias_mm),
                    "thin_by_count": counts < float(min_tube_count_p10),
                }
            )
        )
    return pd.DataFrame(summary_rows), pd.concat(point_rows, ignore_index=True)


def fk_single_u(
    u: np.ndarray,
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    s10: float,
    s20: float,
    ds1: float,
    ds2: float,
    dq: float,
) -> np.ndarray:
    beta, _s1, _s2, _q = beta_from_u_deg(np.asarray(u, dtype=float).reshape(1, 3), s10=s10, s20=s20, ds1=ds1, ds2=ds2, dq=dq)
    theta = theta_from_beta_batch(beta, theta_sign=theta_sign)
    return fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)[0]


def optimize_u_for_target(
    target_xyz: np.ndarray,
    init_u: np.ndarray,
    args: argparse.Namespace,
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
) -> tuple[np.ndarray, float, bool, int]:
    bounds = (np.asarray([-15.0, -15.0, -1.0], dtype=float), np.asarray([15.0, 15.0, 1.0], dtype=float))

    def residual(u: np.ndarray) -> np.ndarray:
        xyz = fk_single_u(
            u,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
            s10=float(args.s10),
            s20=float(args.s20),
            ds1=float(args.ds1),
            ds2=float(args.ds2),
            dq=float(args.dq),
        )
        return (xyz - target_xyz) * 1000.0

    res = least_squares(
        residual,
        np.asarray(init_u, dtype=float),
        bounds=bounds,
        max_nfev=int(args.inverse_max_nfev),
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    err_mm = float(np.linalg.norm(res.fun))
    return res.x.astype(float), err_mm, bool(res.success), int(res.nfev)


def lattice_offsets(args: argparse.Namespace) -> np.ndarray:
    if args.lattice_mode == "pilot":
        da = _parse_float_list(args.pilot_du_deg)
        de = _parse_float_list(args.pilot_deta)
    elif args.lattice_mode == "full":
        da = _parse_float_list(args.full_du_deg)
        de = _parse_float_list(args.full_deta)
    else:
        da = [0.0]
        de = [0.0]
    aa, bb, ee = np.meshgrid(np.asarray(da), np.asarray(da), np.asarray(de), indexing="ij")
    return np.column_stack([aa.ravel(), bb.ravel(), ee.ravel()])


def generate_enrichment(
    base_pool: pd.DataFrame,
    candidates: pd.DataFrame,
    before_points: pd.DataFrame,
    args: argparse.Namespace,
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_xyz = base_pool[XYZ_COLS].to_numpy(dtype=float)
    base_u = base_pool[U_COLS].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=min(int(args.init_k), len(base_pool))).fit(base_xyz)
    offsets = lattice_offsets(args)
    inv_rows: list[dict[str, Any]] = []
    u_rows: list[dict[str, Any]] = []
    for _idx, cand in candidates[candidates["selection_role"].eq("target")].iterrows():
        target, angle = ellipse_points(cand, int(args.n_points))
        cand_points = before_points[(before_points["pool_label"] == "before") & (before_points["candidate_id"] == cand["candidate_id"])].copy()
        thin = cand_points[cand_points[["thin_by_dist", "thin_by_x", "thin_by_count"]].any(axis=1)].copy()
        if int(args.max_thin_points) > 0 and len(thin) > int(args.max_thin_points):
            thin["thin_score"] = (
                np.maximum(thin["nearest_dist_mm"].to_numpy(dtype=float) - float(args.max_nn_p95_mm), 0.0)
                + np.maximum(np.abs(thin["nearest_x_diff_mm"].to_numpy(dtype=float)) - float(args.max_x_bias_mm), 0.0)
                + np.maximum(float(args.min_tube_count_p10) - thin["tube_count_15mm"].to_numpy(dtype=float), 0.0)
            )
            thin = thin.sort_values("thin_score", ascending=False).head(int(args.max_thin_points))
        if thin.empty:
            continue
        _, init_idx = nn.kneighbors(target[thin["point_index"].to_numpy(dtype=np.int64)], n_neighbors=min(int(args.init_k), len(base_pool)))
        for row_pos, point_index in enumerate(thin["point_index"].to_numpy(dtype=np.int64)):
            target_xyz = target[int(point_index)]
            solutions: list[dict[str, Any]] = []
            for seed_rank, base_idx in enumerate(init_idx[row_pos]):
                opt_u, err_mm, ok, nfev = optimize_u_for_target(
                    target_xyz,
                    base_u[int(base_idx)],
                    args,
                    lengths_m=lengths_m,
                    p_end_local_m=p_end_local_m,
                    theta_sign=theta_sign,
                )
                solutions.append(
                    {
                        "candidate_id": cand["candidate_id"],
                        "point_index": int(point_index),
                        "angle_rad": float(angle[int(point_index)]),
                        "seed_rank": int(seed_rank),
                        "seed_row": int(base_idx),
                        "u_a_deg": float(opt_u[0]),
                        "u_b_deg": float(opt_u[1]),
                        "u_eta": float(opt_u[2]),
                        "inverse_residual_mm": err_mm,
                        "inverse_success": bool(ok and err_mm <= float(args.inverse_residual_gate_mm)),
                        "inverse_nfev": int(nfev),
                    }
                )
            good = [s for s in solutions if s["inverse_success"]]
            good = sorted(good, key=lambda x: x["inverse_residual_mm"])
            unique_good: list[dict[str, Any]] = []
            seen: set[tuple[float, float, float]] = set()
            for sol in good:
                key = (round(sol["u_a_deg"], 4), round(sol["u_b_deg"], 4), round(sol["u_eta"], 5))
                if key in seen:
                    continue
                seen.add(key)
                unique_good.append(sol)
                if len(unique_good) >= int(args.solutions_per_point):
                    break
            inv_rows.extend(solutions)
            for sol in unique_good:
                center = np.asarray([sol["u_a_deg"], sol["u_b_deg"], sol["u_eta"]], dtype=float)
                for off in offsets:
                    u = np.clip(center + off, [-15.0, -15.0, -1.0], [15.0, 15.0, 1.0])
                    u_rows.append(
                        {
                            "u_a_deg": float(u[0]),
                            "u_b_deg": float(u[1]),
                            "u_eta": float(u[2]),
                            "enrichment_source_candidate_id": sol["candidate_id"],
                            "enrichment_source_point_index": int(sol["point_index"]),
                            "enrichment_source_angle_rad": float(sol["angle_rad"]),
                            "inverse_residual_mm": float(sol["inverse_residual_mm"]),
                            "sample_kind": f"tube_{args.lattice_mode}",
                        }
                    )
    inv_df = pd.DataFrame(inv_rows)
    if not u_rows:
        return pd.DataFrame(), inv_df
    u_df = pd.DataFrame(u_rows)
    u_df["_ua_round"] = np.round(u_df["u_a_deg"].to_numpy(dtype=float), 5)
    u_df["_ub_round"] = np.round(u_df["u_b_deg"].to_numpy(dtype=float), 5)
    u_df["_ue_round"] = np.round(u_df["u_eta"].to_numpy(dtype=float), 6)
    u_df = u_df.sort_values("inverse_residual_mm").drop_duplicates(["_ua_round", "_ub_round", "_ue_round"]).drop(columns=["_ua_round", "_ub_round", "_ue_round"])
    enrich = attach_fk(
        u_df,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        s10=float(args.s10),
        s20=float(args.s20),
        ds1=float(args.ds1),
        ds2=float(args.ds2),
        dq=float(args.dq),
    )
    enrich["candidate_id"] = "A1_tube_aware_enriched"
    enrich["family"] = "A"
    enrich["base_id"] = "A1"
    enrich["path_name"] = "A1_tube_aware_enriched"
    enrich["source_component"] = "A1_tube_aware_enriched"
    enrich["policy_version"] = "tube_aware_ellipse_enrichment_v1"
    enrich["sample_id"] = np.arange(len(enrich), dtype=np.int64)
    enrich = attach_jacobian(
        enrich,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        s10=float(args.s10),
        s20=float(args.s20),
        ds1=float(args.ds1),
        ds2=float(args.ds2),
        dq=float(args.dq),
        delta=float(args.jacobian_delta),
        sigma3_min_m=float(args.sigma3_min_m),
        kappa_max=float(args.kappa_max),
        batch_size=int(args.jacobian_batch_size),
    )
    return enrich, inv_df


def merge_pool(base_pool: pd.DataFrame, enrich_pass: pd.DataFrame) -> pd.DataFrame:
    base = base_pool.copy().reset_index(drop=True)
    base["sample_kind"] = base.get("sample_kind", "base")
    if enrich_pass.empty:
        return base
    missing = [c for c in base.columns if c not in enrich_pass.columns]
    for col in missing:
        enrich_pass[col] = np.nan
    missing_base = [c for c in enrich_pass.columns if c not in base.columns]
    for col in missing_base:
        base[col] = np.nan
    merged = pd.concat([base[enrich_pass.columns], enrich_pass], ignore_index=True)
    merged["_ua_round"] = np.round(merged["u_a_deg"].to_numpy(dtype=float), 5)
    merged["_ub_round"] = np.round(merged["u_b_deg"].to_numpy(dtype=float), 5)
    merged["_ue_round"] = np.round(merged["u_eta"].to_numpy(dtype=float), 6)
    merged["_priority"] = np.where(merged["sample_kind"].astype(str).str.startswith("tube_"), 0, 1)
    merged = merged.sort_values(["_priority"]).drop_duplicates(["_ua_round", "_ub_round", "_ue_round"], keep="first")
    return merged.drop(columns=["_ua_round", "_ub_round", "_ue_round", "_priority"]).reset_index(drop=True)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_support_by_angle(points: pd.DataFrame, out_path: Path) -> None:
    target = points[points["selection_role"].eq("target")].copy()
    if target.empty:
        return
    cands = list(target["candidate_id"].drop_duplicates())[:3]
    fig, axes = plt.subplots(len(cands), 2, figsize=(12, 3.5 * len(cands)), squeeze=False)
    for row_idx, cid in enumerate(cands):
        part = target[target["candidate_id"].eq(cid)].copy()
        angle_deg = part["angle_rad"].to_numpy(dtype=float) * 180.0 / math.pi
        for label, style in [("before", "--"), ("after", "-")]:
            sub = part[part["pool_label"].eq(label)].sort_values("angle_rad")
            if sub.empty:
                continue
            axes[row_idx, 0].plot(sub["angle_rad"].to_numpy(dtype=float) * 180.0 / math.pi, sub["nearest_dist_mm"], style, label=label)
            axes[row_idx, 1].plot(sub["angle_rad"].to_numpy(dtype=float) * 180.0 / math.pi, sub["tube_count_15mm"], style, label=label)
        axes[row_idx, 0].axhline(8.0, color="#444", lw=0.8)
        axes[row_idx, 1].axhline(16.0, color="#444", lw=0.8)
        axes[row_idx, 0].set_title(f"{cid}: nearest distance")
        axes[row_idx, 1].set_title(f"{cid}: tube count")
        for ax in axes[row_idx]:
            ax.set_xlabel("ellipse angle (deg)")
            ax.legend(loc="best")
    _savefig(fig, out_path)


def plot_summary(before_after: pd.DataFrame, out_path: Path) -> None:
    if before_after.empty:
        return
    labels = before_after["candidate_id"].astype(str) + "\n" + before_after["pool_label"].astype(str)
    x = np.arange(len(before_after))
    fig, ax1 = plt.subplots(figsize=(max(9, 0.5 * len(before_after)), 4.8))
    ax1.bar(x - 0.18, before_after["nn_p95_mm"], width=0.36, label="nn p95 mm", color="#5177a3")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, before_after["tube_count_p10"], width=0.36, label="tube count p10", color="#c46a3a")
    ax1.axhline(8.0, color="#5177a3", lw=0.9, ls="--")
    ax2.axhline(16.0, color="#c46a3a", lw=0.9, ls="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax1.set_ylabel("nn p95 (mm)")
    ax2.set_ylabel("tube count p10")
    ax1.set_title("Tube-aware enrichment support before/after")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    _savefig(fig, out_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.robot_config))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    base_pool = pd.read_parquet(args.base_pool)
    missing = [c for c in XYZ_COLS + BETA_COLS + THETA_COLS + U_COLS if c not in base_pool.columns]
    if missing:
        raise SystemExit(f"base pool missing columns: {missing}")
    candidates = select_candidates(args)
    candidates.to_csv(out_dir / "selected_candidates.csv", index=False)
    before_summary, before_points = support_metrics_for_pool(
        base_pool,
        candidates,
        n_points=int(args.n_points),
        tube_radius_mm=float(args.tube_radius_mm),
        beta_pair_radius_mm=float(args.beta_pair_radius_mm),
        min_tube_count_p10=int(args.min_tube_count_p10),
        beta_gate_deg=float(args.beta_gate_deg),
        max_nn_p95_mm=float(args.max_nn_p95_mm),
        max_x_bias_mm=float(args.max_x_bias_mm),
        pool_label="before",
    )
    enrich_raw, inverse_df = generate_enrichment(
        base_pool,
        candidates,
        before_points,
        args,
        lengths_m=inputs.lengths_m,
        p_end_local_m=inputs.p_end_local_m,
        theta_sign=theta_sign,
    )
    enrich_pass = enrich_raw[enrich_raw["jacobian_gate_pass"].astype(bool)].copy().reset_index(drop=True) if not enrich_raw.empty else enrich_raw
    enriched_pool = merge_pool(base_pool, enrich_pass)
    after_summary, after_points = support_metrics_for_pool(
        enriched_pool,
        candidates,
        n_points=int(args.n_points),
        tube_radius_mm=float(args.tube_radius_mm),
        beta_pair_radius_mm=float(args.beta_pair_radius_mm),
        min_tube_count_p10=int(args.min_tube_count_p10),
        beta_gate_deg=float(args.beta_gate_deg),
        max_nn_p95_mm=float(args.max_nn_p95_mm),
        max_x_bias_mm=float(args.max_x_bias_mm),
        pool_label="after",
    )
    before_after = pd.concat([before_summary, after_summary], ignore_index=True)
    per_angle = pd.concat([before_points, after_points], ignore_index=True)
    before_after.to_csv(out_dir / "candidate_support_before_after.csv", index=False)
    per_angle.to_parquet(out_dir / "per_angle_support.parquet", index=False, compression="zstd")
    inverse_df.to_csv(out_dir / "inverse_refinement_attempts.csv", index=False)
    if not enrich_raw.empty:
        enrich_raw.to_parquet(out_dir / "enriched_candidates_raw.parquet", index=False, compression="zstd")
        enrich_pass.to_parquet(out_dir / "enriched_candidates_jac_pass.parquet", index=False, compression="zstd")
    enriched_pool.to_parquet(out_dir / "enriched_pool.parquet", index=False, compression="zstd")
    plot_support_by_angle(per_angle, out_dir / "support_by_angle.png")
    plot_summary(before_after, out_dir / "tube_count_before_after.png")
    target_after = after_summary[after_summary["selection_role"].eq("target")].copy()
    passed = target_after[target_after["branch_gate_pass"].astype(bool)].copy()
    pilot_gate_pass = bool(not passed.empty)
    report = {
        "mode": "tube_aware_ellipse_enrichment",
        "out_dir": str(out_dir),
        "base_pool": str(args.base_pool),
        "base_rows": int(len(base_pool)),
        "selected_candidates": int(len(candidates)),
        "enriched_raw_rows": int(len(enrich_raw)),
        "enriched_jac_pass_rows": int(len(enrich_pass)),
        "enriched_pool_rows": int(len(enriched_pool)),
        "inverse_attempts": int(len(inverse_df)),
        "inverse_successes": int(inverse_df["inverse_success"].sum()) if not inverse_df.empty else 0,
        "target_after_branch_pass": int(target_after["branch_gate_pass"].sum()) if not target_after.empty else 0,
        "gate_pass": pilot_gate_pass,
        "elapsed_s": float(time.perf_counter() - t0),
        "best_target_after": target_after.sort_values(["branch_gate_pass", "nn_p95_mm", "tube_count_p10"], ascending=[False, True, False]).head(5).to_dict(orient="records"),
    }
    (out_dir / "enrichment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "gate_pass": pilot_gate_pass, "enriched_pool_rows": int(len(enriched_pool))}, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tube-aware local enrichment around large ellipse candidates on the A1 canonical branch.")
    ap.add_argument("--base-pool", type=Path, default=REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "03_improved_u_manifold_sweep_medium_top" / "A1" / "jacobian_pass_pool.parquet")
    ap.add_argument("--candidate-csv", type=Path, default=REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "01_existing_pool_free_ellipse_search" / "path_a_candidates.csv")
    ap.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "11_tube_aware_enrichment_100mm_v1")
    ap.add_argument("--target-amp-xy-mm", type=float, default=100.0)
    ap.add_argument("--control-amp-xy-mm", default="75,87.5")
    ap.add_argument("--max-target-candidates", type=int, default=3)
    ap.add_argument("--max-control-candidates", type=int, default=1)
    ap.add_argument("--n-points", type=int, default=360)
    ap.add_argument("--tube-radius-mm", type=float, default=15.0)
    ap.add_argument("--beta-pair-radius-mm", type=float, default=10.0)
    ap.add_argument("--min-tube-count-p10", type=int, default=16)
    ap.add_argument("--beta-gate-deg", type=float, default=2.0)
    ap.add_argument("--max-nn-p95-mm", type=float, default=8.0)
    ap.add_argument("--max-x-bias-mm", type=float, default=2.0)
    ap.add_argument("--init-k", type=int, default=8)
    ap.add_argument("--solutions-per-point", type=int, default=1)
    ap.add_argument("--max-thin-points", type=int, default=0)
    ap.add_argument("--inverse-residual-gate-mm", type=float, default=3.0)
    ap.add_argument("--inverse-max-nfev", type=int, default=80)
    ap.add_argument("--lattice-mode", choices=["point", "pilot", "full"], default="pilot")
    ap.add_argument("--pilot-du-deg", default="-0.25,0,0.25")
    ap.add_argument("--pilot-deta", default="-0.05,0,0.05")
    ap.add_argument("--full-du-deg", default="-0.4,-0.2,0,0.2,0.4")
    ap.add_argument("--full-deta", default="-0.08,-0.04,0,0.04,0.08")
    ap.add_argument("--s10", type=float, default=0.125)
    ap.add_argument("--s20", type=float, default=0.25)
    ap.add_argument("--ds1", type=float, default=0.075)
    ap.add_argument("--ds2", type=float, default=0.1)
    ap.add_argument("--dq", type=float, default=0.0)
    ap.add_argument("--sigma3-min-m", type=float, default=0.002)
    ap.add_argument("--kappa-max", type=float, default=50.0)
    ap.add_argument("--jacobian-delta", type=float, default=1.0e-3)
    ap.add_argument("--jacobian-batch-size", type=int, default=4096)
    return ap.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
