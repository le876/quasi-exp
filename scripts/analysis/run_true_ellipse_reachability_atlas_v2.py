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
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from true_ellipse_atlas_utils import (  # noqa: E402
    BETA_COLS,
    TARGET_XYZ_COLS,
    THETA_COLS,
    XYZ_COLS,
    beta_bounds_rad,
    cluster_beta_candidates,
    fk_from_beta_batch,
    jacobian_metrics,
    link_cyclic_branch,
    make_axis_phase_ellipse,
    make_normal_tube_targets,
    make_reachability_pool,
    numerical_jacobian_beta,
    solve_beta_ik,
    smoothness_report,
    support_summary_for_targets,
    theta_from_beta_batch,
    trajectory_geometry_metrics,
    weighted_damped_pinv,
    write_json,
    write_markdown_table,
)


DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "true_ellipse_reachability_atlas_v2"
DEFAULT_BASELINE_DIR = REPO_ROOT / "runs" / "true_sinsincos_tube_canonical_v1"
DEFAULT_A1_POOL = REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "03_improved_u_manifold_sweep_medium_top" / "A1" / "jacobian_pass_pool.parquet"
DEFAULT_PATH_D_POOL = REPO_ROOT / "data" / "canonical_layer_field_u3_pilot_v1" / "path_d_wide_redistribute" / "jacobian_pass_pool.parquet"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _phase_selected(args: argparse.Namespace, name: str) -> bool:
    phases = [p.strip() for p in str(args.phases).split(",") if p.strip()]
    return "all" in phases or name in phases


def _load_robot(args: argparse.Namespace) -> tuple[Any, np.ndarray, np.ndarray, float]:
    cfg = load_config(str(args.robot_config))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    return cfg, inputs.lengths_m, inputs.p_end_local_m, theta_sign


def apply_preset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    preset = str(args.preset)
    if preset == "smoke":
        args.pool_rows = min(int(args.pool_rows), 512)
        args.max_centers = min(int(args.max_centers), 4)
        args.phase_step_deg = max(float(args.phase_step_deg), 90.0)
        args.amp_xy_mm = "25"
        args.coarse_points = min(int(args.coarse_points), 12)
        args.refined_points = min(int(args.refined_points), 8)
        args.max_candidates_per_radius = min(int(args.max_candidates_per_radius), 1)
        args.max_pointwise_candidates = min(int(args.max_pointwise_candidates), 1)
        args.max_ik_targets = min(int(args.max_ik_targets), 4)
        args.max_ik_nfev = min(int(args.max_ik_nfev), 20)
        args.max_ik_seeds_per_target = min(int(args.max_ik_seeds_per_target), 2)
        args.init_k = min(int(args.init_k), 3)
        args.max_tube_trajectories = min(int(args.max_tube_trajectories), 1)
        args.max_tube_nfev = min(int(args.max_tube_nfev), 20)
    elif preset == "pilot":
        args.pool_rows = int(args.pool_rows) if int(args.pool_rows) != 1048576 else 1048576
        args.max_centers = int(args.max_centers)
    elif preset == "formal":
        args.pool_rows = max(int(args.pool_rows), 4194304)
        args.max_centers = max(int(args.max_centers), 512)
    else:
        raise ValueError(f"unsupported preset: {preset}")
    return args


def phase0_baseline_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir) / "00_baseline_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = Path(args.baseline_dir)
    support_path = baseline / "02_support_recompute" / "support_summary_by_pool.csv"
    centerline_path = baseline / "03_centerline_active_inverse" / "centerline_inverse_report.csv"
    summary: dict[str, Any] = {
        "mode": "negative_control",
        "baseline_dir": str(baseline),
        "support_path": str(support_path),
        "centerline_path": str(centerline_path),
        "support_exists": support_path.exists(),
        "centerline_exists": centerline_path.exists(),
    }
    support_df = pd.read_csv(support_path) if support_path.exists() else pd.DataFrame()
    center_df = pd.read_csv(centerline_path) if centerline_path.exists() else pd.DataFrame()
    if not support_df.empty:
        summary["support_rows"] = int(len(support_df))
        summary["support_records"] = support_df.to_dict("records")
    if not center_df.empty:
        summary["a1_centerline_residual_p95_mm"] = float(center_df.iloc[0].get("residual_p95_mm", np.nan))
        summary["centerline_records"] = center_df.to_dict("records")
    write_json(out_dir / "current_failure_summary.json", summary)
    lines = [
        "# Current failure summary",
        "",
        "This is the fixed negative control for the V2 reachability atlas experiment.",
        "",
        f"- baseline_dir: `{baseline}`",
        f"- support_exists: `{support_path.exists()}`",
        f"- centerline_exists: `{centerline_path.exists()}`",
        "",
    ]
    if not support_df.empty:
        lines += ["## Support negative control", ""]
        tmp = out_dir / "_support.md"
        write_markdown_table(support_df, tmp, title="support")
        lines += tmp.read_text(encoding="utf-8").splitlines()[2:] + [""]
        tmp.unlink(missing_ok=True)
    if not center_df.empty:
        lines += ["## A1 centerline negative control", ""]
        tmp = out_dir / "_center.md"
        write_markdown_table(center_df, tmp, title="centerline")
        lines += tmp.read_text(encoding="utf-8").splitlines()[2:] + [""]
        tmp.unlink(missing_ok=True)
    (out_dir / "current_failure_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def phase1_reachability_pool(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "01_reachability_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    _cfg, lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    pool = make_reachability_pool(
        n=int(args.pool_rows),
        domain="current",
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        seed=int(args.seed),
        source_domain="current",
    )
    pool_path = out_dir / ("sobol_current_bounds_1m.parquet" if int(args.pool_rows) >= 1_000_000 else "sobol_current_bounds_smoke.parquet")
    pool.to_parquet(pool_path, index=False, compression="zstd")
    pool.to_parquet(out_dir / "reachability_pool_merged.parquet", index=False, compression="zstd")
    report = pd.DataFrame(
        [
            {
                "source_domain": "current",
                "rows": len(pool),
                "x_min_m": float(pool["x_m"].min()),
                "x_max_m": float(pool["x_m"].max()),
                "y_min_m": float(pool["y_m"].min()),
                "y_max_m": float(pool["y_m"].max()),
                "z_min_m": float(pool["z_m"].min()),
                "z_max_m": float(pool["z_m"].max()),
            }
        ]
    )
    report.to_csv(out_dir / "pool_convergence_report.csv", index=False)
    return pool


def _load_pool(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.out_dir) / "01_reachability_pool" / "reachability_pool_merged.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return phase1_reachability_pool(args)


def _choose_centers(pool: pd.DataFrame, *, max_centers: int, voxel_mm: float, seed: int) -> pd.DataFrame:
    xyz = pool[XYZ_COLS].to_numpy(dtype=float)
    vox = np.floor(xyz / (float(voxel_mm) / 1000.0)).astype(np.int64)
    tmp = pd.DataFrame(vox, columns=["vx", "vy", "vz"])
    tmp["row"] = np.arange(len(tmp), dtype=np.int64)
    grouped = tmp.groupby(["vx", "vy", "vz"], sort=False)["row"].agg(["count", "first"]).reset_index()
    centers = pool.iloc[grouped["first"].to_numpy(dtype=np.int64)][XYZ_COLS].copy().reset_index(drop=True)
    centers["voxel_count"] = grouped["count"].to_numpy(dtype=np.int64)
    centers = centers.sort_values("voxel_count", ascending=False).reset_index(drop=True)
    if len(centers) > int(max_centers):
        centers = centers.head(int(max_centers)).copy().reset_index(drop=True)
    centers["center_id"] = [f"c{i:04d}" for i in range(len(centers))]
    return centers


def _support_for_candidate(candidate: pd.DataFrame, pool_xyz: np.ndarray, nn: NearestNeighbors) -> dict[str, Any]:
    target = candidate[TARGET_XYZ_COLS].to_numpy(dtype=float)
    dist, _idx = nn.kneighbors(target)
    nn_mm = dist[:, 0] * 1000.0
    geom = trajectory_geometry_metrics(target)
    row = candidate.iloc[0].to_dict()
    return {
        "candidate_id": row["candidate_id"],
        "center_id": row.get("center_id", ""),
        "center_x_m": row["center_x_m"],
        "center_y_m": row["center_y_m"],
        "center_z_m": row["center_z_m"],
        "amp_xy_mm": row["amp_xy_mm"],
        "amp_z_mm": row["amp_z_mm"],
        "phase_y_rad": row["phase_y_rad"],
        "phase_z_rad": row["phase_z_rad"],
        "nn_p95_mm": float(np.percentile(nn_mm, 95)),
        "nn_max_mm": float(np.max(nn_mm)),
        "empty15_ratio": float(np.mean(nn_mm > 15.0)),
        "geom_score": float(np.percentile(nn_mm, 95) + 0.5 * np.max(nn_mm) + 20.0 * np.mean(nn_mm > 15.0)),
        **geom,
    }


def phase2_ellipse_search(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "02_ellipse_family_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = _load_pool(args)
    pool_xyz = pool[XYZ_COLS].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(pool_xyz)
    centers = _choose_centers(pool, max_centers=int(args.max_centers), voxel_mm=float(args.center_voxel_mm), seed=int(args.seed))
    amps = [float(v.strip()) for v in str(args.amp_xy_mm).split(",") if v.strip()]
    phase_vals = np.deg2rad(np.arange(0.0, 360.0, float(args.phase_step_deg), dtype=float))
    params: list[dict[str, Any]] = []
    for c in centers.itertuples(index=False):
        for amp in amps:
            for py in phase_vals:
                for pz in phase_vals:
                    cid = f"{c.center_id}_a{amp:g}_py{math.degrees(py):.0f}_pz{math.degrees(pz):.0f}".replace(".", "p")
                    params.append(
                        {
                            "candidate_id": cid,
                            "center_id": str(c.center_id),
                            "center_x_m": float(c.x_m),
                            "center_y_m": float(c.y_m),
                            "center_z_m": float(c.z_m),
                            "amp_xy_mm": float(amp),
                            "amp_z_mm": 1.5 * float(amp),
                            "phase_y_rad": float(py),
                            "phase_z_rad": float(pz),
                        }
                    )
    rows: list[dict[str, Any]] = []
    n_points = int(args.coarse_points)
    t = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False, dtype=float)
    sin_t = np.sin(t)
    chunk_size = int(args.search_chunk_candidates)
    for start in range(0, len(params), chunk_size):
        chunk = params[start : start + chunk_size]
        m = len(chunk)
        target = np.empty((m, n_points, 3), dtype=float)
        for i, p in enumerate(chunk):
            amp_xy = float(p["amp_xy_mm"]) / 1000.0
            amp_z = 1.5 * amp_xy
            target[i, :, 0] = float(p["center_x_m"]) + amp_xy * sin_t
            target[i, :, 1] = float(p["center_y_m"]) + amp_xy * np.sin(t + float(p["phase_y_rad"]))
            target[i, :, 2] = float(p["center_z_m"]) + amp_z * np.sin(t + float(p["phase_z_rad"]))
        dist, _idx = nn.kneighbors(target.reshape(-1, 3), n_neighbors=1)
        dist_mm = (dist[:, 0] * 1000.0).reshape(m, n_points)
        for i, p in enumerate(chunk):
            nn_i = dist_mm[i]
            geom = trajectory_geometry_metrics(target[i])
            rows.append(
                {
                    **p,
                    "nn_p95_mm": float(np.percentile(nn_i, 95)),
                    "nn_max_mm": float(np.max(nn_i)),
                    "empty15_ratio": float(np.mean(nn_i > 15.0)),
                    "geom_score": float(np.percentile(nn_i, 95) + 0.5 * np.max(nn_i) + 20.0 * np.mean(nn_i > 15.0)),
                    **geom,
                }
            )
    coarse = pd.DataFrame(rows)
    coarse.to_csv(out_dir / "coarse_candidates.csv", index=False)
    top_frames = []
    for amp, part in coarse[coarse["rank2_gate_pass"].astype(bool)].groupby("amp_xy_mm", sort=False):
        top_frames.append(part.sort_values("geom_score", ascending=True).head(int(args.max_candidates_per_radius)))
    top = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    top.to_csv(out_dir / "top_candidates_by_radius.csv", index=False)
    top.to_csv(out_dir / "refined_candidates.csv", index=False)
    write_markdown_table(top, out_dir / "top_candidates_by_radius.md", title="Top true ellipse candidates by radius")
    if not coarse.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(coarse["amp_xy_mm"], coarse["nn_p95_mm"], s=10, alpha=0.6)
        ax.axhline(8.0, color="#333", ls="--", lw=1)
        ax.set_xlabel("amp_xy (mm)")
        ax.set_ylabel("NN p95 (mm)")
        ax.set_title("Reachability support coarse search")
        fig.tight_layout()
        fig.savefig(out_dir / "coarse_support_scatter.png", dpi=160)
        plt.close(fig)
    return top


def _candidate_targets(row: pd.Series, *, n_points: int) -> pd.DataFrame:
    target = make_axis_phase_ellipse(
        candidate_id=str(row["candidate_id"]),
        center=(float(row["center_x_m"]), float(row["center_y_m"]), float(row["center_z_m"])),
        amp_xy_mm=float(row["amp_xy_mm"]),
        phase_y_rad=float(row["phase_y_rad"]),
        phase_z_rad=float(row["phase_z_rad"]),
        n_points=int(n_points),
    )
    target["center_id"] = row.get("center_id", "")
    return target


def _initial_betas_for_target(target_xyz: np.ndarray, pool: pd.DataFrame, nn: NearestNeighbors, *, k: int, rng: np.random.Generator, bounds: np.ndarray) -> np.ndarray:
    _dist, idx = nn.kneighbors(np.asarray(target_xyz, dtype=float).reshape(1, 3), n_neighbors=min(int(k), len(pool)))
    init = [pool.iloc[i][BETA_COLS].to_numpy(dtype=float) for i in idx[0]]
    for _ in range(4):
        base = init[0]
        init.append(np.clip(base + rng.normal(0.0, np.deg2rad(1.0), size=6), bounds[:, 0], bounds[:, 1]))
    init.append(np.zeros(6, dtype=float))
    return np.vstack(init)


def phase3_pointwise_ik(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "03_fullbeta_pointwise_ik"
    out_dir.mkdir(parents=True, exist_ok=True)
    _cfg, lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    bounds = beta_bounds_rad("current")
    pool = _load_pool(args)
    pool_xyz = pool[XYZ_COLS].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=min(16, len(pool))).fit(pool_xyz)
    top_path = Path(args.out_dir) / "02_ellipse_family_search" / "top_candidates_by_radius.csv"
    top = pd.read_csv(top_path) if top_path.exists() else phase2_ellipse_search(args)
    if str(args.pointwise_amp_mm).strip():
        allowed_amp = {float(v.strip()) for v in str(args.pointwise_amp_mm).split(",") if v.strip()}
        top = top[top["amp_xy_mm"].astype(float).isin(allowed_amp)].copy().reset_index(drop=True)
    if top.empty:
        pd.DataFrame().to_csv(out_dir / "candidate_feasibility_summary.csv", index=False)
        return pd.DataFrame()
    selected_rows = []
    rng = np.random.default_rng(int(args.seed))
    for _, cand in top.sort_values(["amp_xy_mm", "geom_score"], ascending=[False, True]).head(int(args.max_pointwise_candidates)).iterrows():
        cand_dir = out_dir / str(cand["candidate_id"])
        cand_dir.mkdir(parents=True, exist_ok=True)
        report_path = cand_dir / "pointwise_report.json"
        candidates_path = cand_dir / "pointwise_candidates.parquet"
        if bool(args.skip_existing) and report_path.exists() and candidates_path.exists():
            selected_rows.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue
        targets = _candidate_targets(cand, n_points=int(args.refined_points))
        if int(args.max_ik_targets) > 0:
            targets = targets.head(int(args.max_ik_targets)).copy().reset_index(drop=True)
        rows = []
        for _, trow in targets.iterrows():
            target_xyz = trow[TARGET_XYZ_COLS].to_numpy(dtype=float)
            init = _initial_betas_for_target(target_xyz, pool, nn, k=int(args.init_k), rng=rng, bounds=bounds)
            seed_solutions = []
            for seed_idx, seed_beta in enumerate(init[: int(args.max_ik_seeds_per_target)]):
                sol = solve_beta_ik(
                    target_xyz,
                    init_betas=np.asarray(seed_beta, dtype=float).reshape(1, 6),
                    bounds=bounds,
                    lengths_m=lengths_m,
                    p_end_local_m=p_end_local_m,
                    theta_sign=theta_sign,
                    max_nfev=int(args.max_ik_nfev),
                )
                seed_solutions.append((seed_idx, sol))
            for seed_idx, sol in sorted(seed_solutions, key=lambda item: item[1].residual_mm):
                if sol.residual_mm > 5.0 and seed_idx > 0:
                    continue
                rec = trow.to_dict()
                rec.update({"xyz_residual_mm": sol.residual_mm, "ik_success": bool(sol.residual_mm <= 2.0), "seed_rank": int(seed_idx), "inverse_nfev": sol.nfev})
                for i, col in enumerate(BETA_COLS):
                    rec[col] = sol.beta_rad[i]
                theta = theta_from_beta_batch(sol.beta_rad.reshape(1, 6), theta_sign=theta_sign)[0]
                xyz = sol.xyz_m
                for i, col in enumerate(THETA_COLS):
                    rec[col] = theta[i]
                for i, col in enumerate(XYZ_COLS):
                    rec[col] = xyz[i]
                try:
                    jm = jacobian_metrics(numerical_jacobian_beta(sol.beta_rad, lengths_m=lengths_m, p_end_local_m=p_end_local_m, theta_sign=theta_sign))
                    rec.update(jm)
                except Exception:
                    rec.update({"sigma1_m": np.nan, "sigma2_m": np.nan, "sigma3_m": np.nan, "kappa": np.nan})
                rows.append(rec)
        df = pd.DataFrame(rows)
        clustered = []
        for angle_idx, part in df.groupby("angle_idx", sort=False):
            clustered.append(cluster_beta_candidates(part, max_clusters=16))
        cand_candidates = pd.concat(clustered, ignore_index=True) if clustered else df
        cand_candidates.to_parquet(cand_dir / "pointwise_candidates.parquet", index=False, compression="zstd")
        if not cand_candidates.empty:
            best_by_target = cand_candidates.groupby("angle_idx", sort=False)["xyz_residual_mm"].min().to_numpy(dtype=float)
        else:
            best_by_target = np.asarray([np.inf])
        report = {
            "candidate_id": str(cand["candidate_id"]),
            "rows": int(len(cand_candidates)),
            "target_rows": int(len(targets)),
            "success_ratio": float(np.mean(best_by_target <= 2.0)),
            "residual_p95_mm": float(np.percentile(best_by_target, 95)),
            "residual_max_mm": float(np.max(best_by_target)),
            "pointwise_gate_pass": bool(np.mean(best_by_target <= 2.0) >= 1.0 and np.percentile(best_by_target, 95) <= 2.0 and np.max(best_by_target) <= 5.0),
        }
        write_json(cand_dir / "pointwise_report.json", report)
        selected_rows.append(report)
    summary = pd.DataFrame(selected_rows)
    summary.to_csv(out_dir / "candidate_feasibility_summary.csv", index=False)
    return summary


def phase4_branch_linking(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "04_cyclic_branch_linking"
    out_dir.mkdir(parents=True, exist_ok=True)
    p3 = Path(args.out_dir) / "03_fullbeta_pointwise_ik"
    summary_path = p3 / "candidate_feasibility_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else phase3_pointwise_ik(args)
    rows = []
    for _, row in summary.iterrows():
        cid = str(row["candidate_id"])
        cand_path = p3 / cid / "pointwise_candidates.parquet"
        if not cand_path.exists():
            continue
        candidates = pd.read_parquet(cand_path)
        selected, report = link_cyclic_branch(candidates, max_edge_deg=float(args.branch_max_edge_deg))
        cdir = out_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        if not selected.empty:
            selected.to_parquet(cdir / "linked_branch.parquet", index=False, compression="zstd")
        write_json(cdir / "linking_report.json", report)
        rows.append(
            {
                "candidate_id": cid,
                "success": bool(report.get("success", False)),
                "delta_beta_rms_p95_deg": float(report.get("delta_beta_rms_p95_deg", np.inf)),
                "delta_beta_rms_max_deg": float(report.get("delta_beta_rms_max_deg", np.inf)),
                "seam_beta_rms_deg": float(report.get("seam_beta_rms_deg", np.inf)),
                "zero_reconfiguration": bool(report.get("zero_reconfiguration", False)),
                "reason": str(report.get("reason", "")),
                **{k: v for k, v in report.items() if k not in {"success", "delta_beta_rms_p95_deg", "delta_beta_rms_max_deg", "seam_beta_rms_deg", "zero_reconfiguration", "reason"}},
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "branch_candidate_summary.csv", index=False)
    return out


def phase5_centerline_optimization(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "05_centerline_trajectory_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    branch_summary_path = Path(args.out_dir) / "04_cyclic_branch_linking" / "branch_candidate_summary.csv"
    branch_summary = pd.read_csv(branch_summary_path) if branch_summary_path.exists() else phase4_branch_linking(args)
    empty_cols = [
        "candidate_id",
        "rows",
        "residual_p95_mm",
        "residual_max_mm",
        "delta_beta_p95_deg",
        "delta_beta_max_deg",
        "delta2_beta_p95_deg",
        "seam_beta_rms_deg",
        "kappa_p95",
        "centerline_gate_pass",
    ]
    if branch_summary.empty or "success" not in branch_summary.columns:
        empty = pd.DataFrame(columns=empty_cols)
        empty.to_csv(out_dir / "centerline_final_summary.csv", index=False)
        return empty
    if "delta_beta_rms_p95_deg" not in branch_summary.columns:
        branch_summary["delta_beta_rms_p95_deg"] = np.inf
    rows = []
    for _, row in branch_summary.sort_values(["success", "delta_beta_rms_p95_deg"], ascending=[False, True]).iterrows():
        if not bool(row.get("success", False)):
            continue
        cid = str(row["candidate_id"])
        src = Path(args.out_dir) / "04_cyclic_branch_linking" / cid / "linked_branch.parquet"
        if not src.exists():
            continue
        df = pd.read_parquet(src)
        report = smoothness_report(df)
        report["centerline_gate_pass"] = bool(
            report.get("residual_p95_mm", np.inf) <= 2.0
            and report.get("residual_max_mm", np.inf) <= 5.0
            and report.get("delta_beta_p95_deg", np.inf) <= 0.75
            and report.get("delta2_beta_p95_deg", np.inf) <= 0.25
            and report.get("seam_beta_rms_deg", np.inf) <= 0.75
            and report.get("kappa_p95", 0.0) <= 150.0
        )
        cdir = out_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cdir / "optimized_centerline.parquet", index=False, compression="zstd")
        write_json(cdir / "optimization_report.json", report)
        rows.append({"candidate_id": cid, **report})
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=empty_cols)
    out.to_csv(out_dir / "centerline_final_summary.csv", index=False)
    return out


def phase6_local_atlas_tube(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "06_local_atlas_tube_generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    _cfg, lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    bounds = beta_bounds_rad("current")
    summary_path = Path(args.out_dir) / "05_centerline_trajectory_optimization" / "centerline_final_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else phase5_centerline_optimization(args)
    passed = summary[summary.get("centerline_gate_pass", False).astype(bool)] if not summary.empty and "centerline_gate_pass" in summary.columns else pd.DataFrame()
    reports = []
    manifest_cols = [
        "candidate_id",
        "rows",
        "residual_p95_mm",
        "residual_max_mm",
        "delta_beta_p95_deg",
        "target_success_ratio",
        "tube_gate_pass",
    ]
    if passed.empty:
        write_json(out_dir / "tube_quality_report.json", {"tube_gate_pass": False, "reason": "no_centerline_passed"})
        empty = pd.DataFrame(columns=manifest_cols)
        empty.to_csv(out_dir / "atlas_manifest.csv", index=False)
        return empty
    for _, row in passed.head(int(args.max_tube_trajectories)).iterrows():
        cid = str(row["candidate_id"])
        center = pd.read_parquet(Path(args.out_dir) / "05_centerline_trajectory_optimization" / cid / "optimized_centerline.parquet")
        center_targets = center.copy()
        for a, b in zip(TARGET_XYZ_COLS, XYZ_COLS):
            center_targets[a] = center_targets[b].to_numpy(dtype=float)
        tube_targets = make_normal_tube_targets(center_targets, offsets_mm=[-5, -2.5, 0, 2.5, 5])
        rows = []
        for _, trow in tube_targets.iterrows():
            center_row = center[center["angle_idx"].eq(trow["angle_idx"])].iloc[0]
            beta0 = center_row[BETA_COLS].to_numpy(dtype=float)
            jac = numerical_jacobian_beta(beta0, lengths_m=lengths_m, p_end_local_m=p_end_local_m, theta_sign=theta_sign)
            pred = beta0 + weighted_damped_pinv(jac) @ (trow[TARGET_XYZ_COLS].to_numpy(dtype=float) - center_row[XYZ_COLS].to_numpy(dtype=float))
            sol = solve_beta_ik(
                trow[TARGET_XYZ_COLS].to_numpy(dtype=float),
                init_betas=np.vstack([pred, beta0]),
                bounds=bounds,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
                max_nfev=int(args.max_tube_nfev),
                center_beta=pred,
                lambda_center=0.05,
            )
            rec = trow.to_dict()
            rec.update({"xyz_residual_mm": sol.residual_mm, "tube_success": bool(sol.residual_mm <= 1.5)})
            for i, col in enumerate(BETA_COLS):
                rec[col] = sol.beta_rad[i]
            theta = theta_from_beta_batch(sol.beta_rad.reshape(1, 6), theta_sign=theta_sign)[0]
            for i, col in enumerate(THETA_COLS):
                rec[col] = theta[i]
            for i, col in enumerate(XYZ_COLS):
                rec[col] = sol.xyz_m[i]
            rows.append(rec)
        tube = pd.DataFrame(rows)
        tube.to_parquet(out_dir / f"{cid}_tube_small.parquet", index=False, compression="zstd")
        q = smoothness_report(tube)
        q["candidate_id"] = cid
        q["target_success_ratio"] = float(tube["tube_success"].mean()) if not tube.empty else 0.0
        q["tube_gate_pass"] = bool(q["target_success_ratio"] >= 0.99 and q.get("residual_p95_mm", np.inf) <= 1.5 and q.get("residual_max_mm", np.inf) <= 3.0 and q.get("delta_beta_p95_deg", np.inf) <= 1.0)
        reports.append(q)
    out = pd.DataFrame(reports)
    if out.empty:
        out = pd.DataFrame(columns=manifest_cols)
    out.to_csv(out_dir / "atlas_manifest.csv", index=False)
    write_json(out_dir / "tube_quality_report.json", {"reports": reports, "any_tube_gate_pass": bool(out["tube_gate_pass"].any()) if not out.empty else False})
    return out


def phase10_failure_diagnostics(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / "10_failure_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    p3 = Path(args.out_dir) / "03_fullbeta_pointwise_ik"
    branch_path = Path(args.out_dir) / "04_cyclic_branch_linking" / "branch_candidate_summary.csv"
    rows = []
    threshold_rows = []
    for cand_path in sorted(p3.glob("*/pointwise_candidates.parquet")):
        cid = cand_path.parent.name
        df = pd.read_parquet(cand_path)
        if df.empty:
            continue
        best_idx = df.groupby("angle_idx", sort=True)["xyz_residual_mm"].idxmin().to_numpy(dtype=np.int64)
        best = df.loc[best_idx].sort_values("angle_idx").reset_index(drop=True)
        report = smoothness_report(best)
        report["candidate_id"] = cid
        rows.append(report)
        for thresh in (1.5, 3.0, 5.0, 10.0, 20.0):
            selected, link_report = link_cyclic_branch(df, max_edge_deg=thresh)
            threshold_rows.append(
                {
                    "candidate_id": cid,
                    "threshold_deg": float(thresh),
                    "success": bool(link_report.get("success", False)),
                    "delta_beta_rms_p95_deg": float(link_report.get("delta_beta_rms_p95_deg", np.inf)),
                    "delta_beta_rms_max_deg": float(link_report.get("delta_beta_rms_max_deg", np.inf)),
                    "seam_beta_rms_deg": float(link_report.get("seam_beta_rms_deg", np.inf)),
                    "zero_reconfiguration": bool(link_report.get("zero_reconfiguration", False)),
                    "selected_rows": int(len(selected)),
                    "reason": str(link_report.get("reason", "")),
                }
            )
    best_path_df = pd.DataFrame(rows)
    sweep_df = pd.DataFrame(threshold_rows)
    best_path_df.to_csv(out_dir / "best_residual_branch_smoothness.csv", index=False)
    sweep_df.to_csv(out_dir / "branch_threshold_sweep.csv", index=False)
    branch_df = pd.read_csv(branch_path) if branch_path.exists() else pd.DataFrame()
    lines = [
        "# Failure diagnostics",
        "",
        "Phase 3 showed pointwise full-beta reachability, but Phase 4 did not find a closed continuous branch under the pilot edge gate.",
        "",
    ]
    if not branch_df.empty:
        lines += ["## Branch linking summary", ""]
        tmp = out_dir / "_branch.md"
        write_markdown_table(branch_df, tmp, title="branch")
        lines += tmp.read_text(encoding="utf-8").splitlines()[2:] + [""]
        tmp.unlink(missing_ok=True)
    if not best_path_df.empty:
        lines += ["## Best residual path smoothness", ""]
        tmp = out_dir / "_best.md"
        write_markdown_table(best_path_df, tmp, title="best residual path")
        lines += tmp.read_text(encoding="utf-8").splitlines()[2:] + [""]
        tmp.unlink(missing_ok=True)
    if not sweep_df.empty:
        lines += ["## Threshold sweep", ""]
        tmp = out_dir / "_sweep.md"
        write_markdown_table(sweep_df, tmp, title="threshold sweep")
        lines += tmp.read_text(encoding="utf-8").splitlines()[2:] + [""]
        tmp.unlink(missing_ok=True)
    lines += [
        "## Decision",
        "",
        "Do not generate tube or train models unless a candidate passes the cyclic branch gate. The next experiment should add more IK seeds/candidates, use trajectory-level optimization, or search another ellipse family/center.",
        "",
    ]
    (out_dir / "failure_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def phase11_summary(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / "11_paper_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "baseline": Path(args.out_dir) / "00_baseline_audit" / "current_failure_summary.md",
        "ellipse_search": Path(args.out_dir) / "02_ellipse_family_search" / "top_candidates_by_radius.csv",
        "pointwise": Path(args.out_dir) / "03_fullbeta_pointwise_ik" / "candidate_feasibility_summary.csv",
        "branch": Path(args.out_dir) / "04_cyclic_branch_linking" / "branch_candidate_summary.csv",
        "centerline": Path(args.out_dir) / "05_centerline_trajectory_optimization" / "centerline_final_summary.csv",
        "tube": Path(args.out_dir) / "06_local_atlas_tube_generation" / "atlas_manifest.csv",
        "failure_diagnostics": Path(args.out_dir) / "10_failure_diagnostics" / "branch_threshold_sweep.csv",
    }
    lines = ["# True ellipse reachability atlas V2 summary", "", "本实验先验证 full beta6 可达性、连续闭环 branch 和 local tube，再决定是否训练模型。", ""]
    for key, path in files.items():
        lines += [f"## {key}", "", f"- file: `{path}`", f"- exists: `{path.exists()}`", ""]
        if path.exists() and path.suffix == ".csv":
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            if not df.empty:
                tmp = out_dir / f"_{key}.md"
                write_markdown_table(df.head(20), tmp, title=key)
                lines += tmp.read_text(encoding="utf-8").splitlines()[2:] + [""]
                tmp.unlink(missing_ok=True)
    (out_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    apply_preset_defaults(args)
    t0 = time.perf_counter()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if _phase_selected(args, "phase0"):
        phase0_baseline_audit(args)
    if _phase_selected(args, "phase1"):
        phase1_reachability_pool(args)
    if _phase_selected(args, "phase2"):
        phase2_ellipse_search(args)
    if _phase_selected(args, "phase3"):
        phase3_pointwise_ik(args)
    if _phase_selected(args, "phase4"):
        phase4_branch_linking(args)
    if _phase_selected(args, "phase5"):
        phase5_centerline_optimization(args)
    if _phase_selected(args, "phase6"):
        phase6_local_atlas_tube(args)
    if _phase_selected(args, "phase10"):
        phase10_failure_diagnostics(args)
    if _phase_selected(args, "phase11") or _phase_selected(args, "all"):
        phase11_summary(args)
    payload = {"out_dir": str(args.out_dir), "preset": str(args.preset), "phases": str(args.phases), "elapsed_s": float(time.perf_counter() - t0)}
    write_json(Path(args.out_dir) / "run_report.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run true ellipse global reachability and canonical atlas experiment V2.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    ap.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--preset", choices=["smoke", "pilot", "formal"], default="pilot")
    ap.add_argument("--phases", default="all")
    ap.add_argument("--pool-rows", type=int, default=1048576)
    ap.add_argument("--max-centers", type=int, default=512)
    ap.add_argument("--center-voxel-mm", type=float, default=20.0)
    ap.add_argument("--amp-xy-mm", default="50,75,100")
    ap.add_argument("--phase-step-deg", type=float, default=30.0)
    ap.add_argument("--coarse-points", type=int, default=36)
    ap.add_argument("--refined-points", type=int, default=72)
    ap.add_argument("--max-candidates-per-radius", type=int, default=20)
    ap.add_argument("--search-chunk-candidates", type=int, default=4096)
    ap.add_argument("--max-pointwise-candidates", type=int, default=20)
    ap.add_argument("--pointwise-amp-mm", default="")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--max-ik-targets", type=int, default=0)
    ap.add_argument("--init-k", type=int, default=12)
    ap.add_argument("--max-ik-nfev", type=int, default=100)
    ap.add_argument("--max-ik-seeds-per-target", type=int, default=4)
    ap.add_argument("--branch-max-edge-deg", type=float, default=3.0)
    ap.add_argument("--max-tube-trajectories", type=int, default=2)
    ap.add_argument("--max-tube-nfev", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260710)
    return ap.parse_args(argv)


def main() -> int:
    payload = run(parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
