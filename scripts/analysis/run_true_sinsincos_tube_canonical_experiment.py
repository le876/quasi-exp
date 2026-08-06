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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from true_sinsincos_tube_utils import (  # noqa: E402
    BETA_COLS,
    TARGET_XYZ_COLS,
    THETA_COLS,
    U_DEG_COLS,
    attach_fk_from_u,
    infer_u_mapping_params,
    make_true_sinsincos_ellipse,
    make_tube_targets,
    optimize_u_for_target,
    recompute_support_for_targets,
    trajectory_geometry_metrics,
    write_markdown_table,
)


DEFAULT_A1_POOL = REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "03_improved_u_manifold_sweep_medium_top" / "A1" / "jacobian_pass_pool.parquet"
DEFAULT_HIER_POOL = REPO_ROOT / "data" / "hierarchical_beta_fk_x1p0_1p2_v1" / "x1p0_1p2_pool.parquet"
DEFAULT_ENRICHED_POOL = REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "11_tube_aware_enrichment_100mm_v1" / "enriched_pool.parquet"
DEFAULT_13B_POINTS = REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "13b_final_ellipse_benchmark_tube100_true_sinsincos_v1" / "trajectory_points.parquet"
DEFAULT_13B_SCOREBOARD = REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "13b_final_ellipse_benchmark_tube100_true_sinsincos_v1" / "global_showcase_scoreboard.csv"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _read_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if nrows is not None:
        return pd.read_parquet(path).head(int(nrows))
    return pd.read_parquet(path)


def _phase_selected(args: argparse.Namespace, name: str) -> bool:
    phases = [p.strip() for p in str(args.phases).split(",") if p.strip()]
    return "all" in phases or name in phases


def phase0_input_audit(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / "00_input_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "a1_pool": Path(args.a1_pool),
        "hierarchical_pool": Path(args.hierarchical_pool),
        "old_enriched_pool": Path(args.enriched_pool),
        "old_13b_trajectory_points": Path(args.old_trajectory_points),
        "old_13b_scoreboard": Path(args.old_scoreboard),
    }
    required = ["x_m", "y_m", "z_m", *BETA_COLS, *THETA_COLS]
    u_cols = ["u_a_rad", "u_b_rad", "u_eta", "u_a_deg", "u_b_deg", "s1", "s2"]
    jac_cols = ["sigma1_m", "sigma2_m", "sigma3_m", "kappa"]
    rows: list[dict[str, Any]] = []
    lines = ["# Input Schema Check", ""]
    dup_rows: list[dict[str, Any]] = []
    for key, path in paths.items():
        row: dict[str, Any] = {"dataset_id": key, "path": str(path), "exists": path.exists()}
        if path.exists():
            df = _read_table(path)
            row.update({"rows": int(len(df)), "columns": int(len(df.columns))})
            row["missing_xyz_beta_theta"] = ",".join([c for c in required if c not in df.columns])
            row["missing_u"] = ",".join([c for c in u_cols if c not in df.columns])
            row["missing_jacobian"] = ",".join([c for c in jac_cols if c not in df.columns])
            if "trajectory_family" in df.columns:
                row["trajectory_family_values"] = ",".join(sorted(map(str, df["trajectory_family"].dropna().unique().tolist())))
            if {"model_id", "candidate_id", "angle_rad"}.issubset(df.columns):
                duplicates = int(df.duplicated(["model_id", "candidate_id", "angle_rad"]).sum())
                row["duplicate_model_candidate_angle"] = duplicates
                if duplicates:
                    dup = (
                        df.groupby(["model_id", "candidate_id", "angle_rad"], dropna=False)
                        .size()
                        .reset_index(name="count")
                        .query("count > 1")
                    )
                    dup["dataset_id"] = key
                    dup_rows.extend(dup.to_dict("records"))
        rows.append(row)
        lines += [
            f"## {key}",
            "",
            f"- exists: `{row['exists']}`",
            f"- rows: `{row.get('rows', 0)}`",
            f"- missing xyz/beta/theta: `{row.get('missing_xyz_beta_theta', 'file_missing')}`",
            f"- missing u: `{row.get('missing_u', 'file_missing')}`",
            f"- missing jacobian: `{row.get('missing_jacobian', 'file_missing')}`",
            "",
        ]
    pd.DataFrame(rows).to_csv(out_dir / "input_manifest.csv", index=False)
    (out_dir / "schema_check.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(dup_rows).to_csv(out_dir / "trajectory_duplicate_report.csv", index=False)


def _candidate_centers() -> list[dict[str, Any]]:
    return [
        {
            "ellipse_id": "E75_path_a_known",
            "center": (1.0896, 0.2071, 0.2873),
            "amp_xy_mm": 75.0,
            "phase_rad": math.pi / 2.0,
        },
        {
            "ellipse_id": "E87p5_path_a_known",
            "center": (1.0854, 0.2951, 0.1754),
            "amp_xy_mm": 87.5,
            "phase_rad": 3.0 * math.pi / 2.0,
        },
        {
            "ellipse_id": "E100_path_a_known",
            "center": (1.0904, 0.0507, 0.3322),
            "amp_xy_mm": 100.0,
            "phase_rad": math.pi / 2.0,
        },
    ]


def phase1_generate_targets(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "01_true_trajectory_generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed = {v.strip() for v in str(args.ellipses).split(",") if v.strip()}
    frames: list[pd.DataFrame] = []
    report_rows: list[dict[str, Any]] = []
    for cand in _candidate_centers():
        if allowed and not any(str(cand["ellipse_id"]).startswith(v) for v in allowed):
            continue
        target = make_true_sinsincos_ellipse(
            ellipse_id=str(cand["ellipse_id"]),
            center=tuple(cand["center"]),
            amp_xy_mm=float(cand["amp_xy_mm"]),
            n_points=int(args.n_points),
            phase_rad=float(cand["phase_rad"]),
            include_endpoint=True,
        )
        metrics = trajectory_geometry_metrics(target[TARGET_XYZ_COLS].to_numpy(dtype=float))
        row = {
            "ellipse_id": cand["ellipse_id"],
            "amp_xy_mm": cand["amp_xy_mm"],
            "amp_z_mm": 1.5 * float(cand["amp_xy_mm"]),
            "center_x": cand["center"][0],
            "center_y": cand["center"][1],
            "center_z": cand["center"][2],
            **metrics,
        }
        report_rows.append(row)
        frames.append(target)
    targets = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    targets.to_parquet(out_dir / "true_ellipse_targets.parquet", index=False, compression="zstd")
    report = pd.DataFrame(report_rows)
    report.to_csv(out_dir / "trajectory_geometry_report.csv", index=False)
    write_markdown_table(report, out_dir / "trajectory_svd_report.md", title="True sin_sin_cos trajectory SVD report")
    return targets


def _load_targets(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.out_dir) / "01_true_trajectory_generation" / "true_ellipse_targets.parquet"
    if not path.exists():
        return phase1_generate_targets(args)
    return pd.read_parquet(path)


def _filter_targets_by_requested_ellipses(targets: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    allowed = [v.strip() for v in str(args.ellipses).split(",") if v.strip()]
    if not allowed or "ellipse_id" not in targets.columns:
        return targets
    mask = pd.Series(False, index=targets.index)
    vals = targets["ellipse_id"].astype(str)
    for prefix in allowed:
        mask = mask | vals.str.startswith(prefix)
    return targets.loc[mask].reset_index(drop=True)


def phase2_support(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / "02_support_recompute"
    plot_dir = out_dir / "support_by_angle_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    targets = _load_targets(args)
    pools = {
        "a1": Path(args.a1_pool),
        "old_enriched": Path(args.enriched_pool),
        "hierarchical": Path(args.hierarchical_pool),
    }
    enabled = {v.strip() for v in str(args.support_pools).split(",") if v.strip()}
    summaries: list[pd.DataFrame] = []
    per_angles: list[pd.DataFrame] = []
    for label, path in pools.items():
        if label not in enabled:
            continue
        pool = pd.read_parquet(path)
        summary, per_angle = recompute_support_for_targets(targets, pool, pool_label=label)
        summaries.append(summary)
        per_angles.append(per_angle)
    support_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    per_angle_all = pd.concat(per_angles, ignore_index=True) if per_angles else pd.DataFrame()
    support_summary.to_csv(out_dir / "support_summary_by_pool.csv", index=False)
    per_angle_all.to_parquet(out_dir / "per_angle_support.parquet", index=False, compression="zstd")
    for (ellipse_id, pool_label), part in per_angle_all.groupby(["ellipse_id", "pool_label"], sort=False):
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        angle = part["angle_rad"].to_numpy(dtype=float) * 180.0 / math.pi
        ax1.plot(angle, part["true_nn_mm"], label="NN distance mm", color="#1f4e79")
        ax1.axhline(8.0, lw=0.8, ls="--", color="#1f4e79")
        ax1.set_xlabel("angle (deg)")
        ax1.set_ylabel("NN distance (mm)")
        ax2 = ax1.twinx()
        ax2.plot(angle, part["true_tube_count_15mm"], label="tube count", color="#c4511c")
        ax2.axhline(16.0, lw=0.8, ls="--", color="#c4511c")
        ax2.set_ylabel("tube count within 15mm")
        ax1.set_title(f"{ellipse_id} / {pool_label}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{ellipse_id}_{pool_label}_support.png", dpi=160)
        plt.close(fig)


def _nearest_a1_u(a1_pool: pd.DataFrame, target_xyz: np.ndarray, *, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=min(int(k), len(a1_pool))).fit(a1_pool[["x_m", "y_m", "z_m"]].to_numpy(dtype=float))
    dist, idx = nn.kneighbors(np.asarray(target_xyz, dtype=float).reshape(-1, 3))
    u = a1_pool.iloc[idx[:, 0]][U_DEG_COLS].to_numpy(dtype=float)
    return u, dist[:, 0] * 1000.0, idx[:, 0]


def _active_inverse_targets(
    targets: pd.DataFrame,
    a1_pool: pd.DataFrame,
    args: argparse.Namespace,
    *,
    inputs: Any,
    theta_sign: float,
    mapping: dict[str, float],
    centerline: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_xyz = targets[TARGET_XYZ_COLS].to_numpy(dtype=float)
    nearest_u, nearest_mm, nearest_idx = _nearest_a1_u(a1_pool, target_xyz, k=int(args.init_k))
    rows: list[dict[str, Any]] = []
    selected_u: list[np.ndarray] = []
    selected_meta: list[dict[str, Any]] = []
    prev_u: np.ndarray | None = None
    for i, row in targets.reset_index(drop=True).iterrows():
        init_candidates = [nearest_u[i]]
        if centerline is not None and "u_a_deg" in centerline.columns:
            same_angle = centerline[centerline["angle_idx"].eq(row["angle_idx"])]
            if not same_angle.empty:
                init_candidates.insert(0, same_angle.iloc[0][U_DEG_COLS].to_numpy(dtype=float))
        if prev_u is not None:
            init_candidates.insert(0, prev_u)
        best: dict[str, Any] | None = None
        for seed_rank, init_u in enumerate(init_candidates[: int(args.init_k)]):
            opt_u, err_mm, ok, nfev = optimize_u_for_target(
                row[TARGET_XYZ_COLS].to_numpy(dtype=float),
                np.asarray(init_u, dtype=float),
                fk_func=fk_dh_batch,
                lengths_m=inputs.lengths_m,
                p_end_local_m=inputs.p_end_local_m,
                theta_sign=theta_sign,
                mapping=mapping,
                center_u_deg=np.asarray(init_candidates[0], dtype=float),
                max_nfev=int(args.inverse_max_nfev),
                lambda_u=float(args.inverse_lambda_u),
            )
            rec = row.to_dict()
            rec.update(
                {
                    "candidate_id": seed_rank,
                    "source_type": "active_inverse",
                    "u_a_deg": float(opt_u[0]),
                    "u_b_deg": float(opt_u[1]),
                    "u_eta": float(opt_u[2]),
                    "xyz_residual_mm": float(err_mm),
                    "inverse_success": bool(ok and err_mm <= float(args.inverse_residual_gate_mm)),
                    "inverse_nfev": int(nfev),
                    "nearest_a1_mm": float(nearest_mm[i]),
                    "nearest_a1_row": int(nearest_idx[i]),
                }
            )
            rows.append(rec)
            if best is None or rec["xyz_residual_mm"] < best["xyz_residual_mm"]:
                best = rec
        assert best is not None
        prev_u = np.asarray([best["u_a_deg"], best["u_b_deg"], best["u_eta"]], dtype=float)
        selected_u.append(prev_u)
        selected_meta.append(best)
    candidates = pd.DataFrame(rows)
    selected = pd.DataFrame(selected_meta)
    selected_fk = attach_fk_from_u(
        selected[U_DEG_COLS],
        fk_func=fk_dh_batch,
        lengths_m=inputs.lengths_m,
        p_end_local_m=inputs.p_end_local_m,
        theta_sign=theta_sign,
        mapping=mapping,
    )
    for col in selected.columns:
        if col not in selected_fk.columns:
            selected_fk[col] = selected[col].to_numpy()
    for axis in "xyz":
        selected_fk[f"{axis}_target_m"] = selected[f"{axis}_target_m"].to_numpy(dtype=float)
        selected_fk[f"{axis}_fk_m"] = selected_fk[f"{axis}_m"].to_numpy(dtype=float)
        selected_fk[f"{axis}_m"] = selected[f"{axis}_target_m"].to_numpy(dtype=float)
    selected_fk["sample_id"] = np.arange(len(selected_fk), dtype=np.int64)
    selected_fk["sample_kind"] = "centerline" if centerline is None else "tube"
    return candidates, selected_fk


def _quality_report(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0}
    beta = df[BETA_COLS].to_numpy(dtype=float) if all(c in df.columns for c in BETA_COLS) else np.empty((0, 6))
    delta_beta = np.sqrt(np.mean(np.square(np.diff(beta, axis=0)), axis=1)) * 180.0 / math.pi if len(beta) > 1 else np.array([0.0])
    return {
        "rows": int(len(df)),
        "residual_p95_mm": float(np.percentile(df["xyz_residual_mm"], 95)) if "xyz_residual_mm" in df.columns else None,
        "residual_max_mm": float(np.max(df["xyz_residual_mm"])) if "xyz_residual_mm" in df.columns else None,
        "delta_beta_p95_deg": float(np.percentile(delta_beta, 95)),
        "kappa_p95": float(np.percentile(df["kappa"], 95)) if "kappa" in df.columns else None,
    }


def phase3_centerline(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "03_centerline_active_inverse"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.robot_config))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    a1_pool = pd.read_parquet(args.a1_pool)
    mapping = infer_u_mapping_params(a1_pool)
    targets = _filter_targets_by_requested_ellipses(_load_targets(args), args)
    if int(args.max_angle_points) > 0:
        targets = targets.groupby("ellipse_id", group_keys=False).head(int(args.max_angle_points)).reset_index(drop=True)
    candidates, selected = _active_inverse_targets(targets, a1_pool, args, inputs=inputs, theta_sign=theta_sign, mapping=mapping)
    candidates.to_parquet(out_dir / "centerline_candidates.parquet", index=False, compression="zstd")
    selected.to_parquet(out_dir / "centerline_selected.parquet", index=False, compression="zstd")
    report = pd.DataFrame([{"mapping": json.dumps(mapping), **_quality_report(selected)}])
    report.to_csv(out_dir / "centerline_inverse_report.csv", index=False)
    return selected


def _load_centerline(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.out_dir) / "03_centerline_active_inverse" / "centerline_selected.parquet"
    if not path.exists():
        return phase3_centerline(args)
    return pd.read_parquet(path)


def phase5_tube_targets(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "05_tube_target_generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    centerline = _load_centerline(args)
    center_targets = centerline.copy()
    center_targets["x_target_m"] = center_targets["x_m"].to_numpy(dtype=float)
    center_targets["y_target_m"] = center_targets["y_m"].to_numpy(dtype=float)
    center_targets["z_target_m"] = center_targets["z_m"].to_numpy(dtype=float)
    small = make_tube_targets(center_targets, offsets_mm=[-5.0, 0.0, 5.0])
    full = make_tube_targets(center_targets, offsets_mm=[-10.0, -5.0, 0.0, 5.0, 10.0])
    small.to_parquet(out_dir / "tube_targets_pilot.parquet", index=False, compression="zstd")
    full.to_parquet(out_dir / "tube_targets_full.parquet", index=False, compression="zstd")
    pd.DataFrame(
        [
            {"dataset": "pilot", "rows": len(small), "offsets": "-5,0,5"},
            {"dataset": "full", "rows": len(full), "offsets": "-10,-5,0,5,10"},
        ]
    ).to_csv(out_dir / "tube_frame_report.csv", index=False)
    return small if str(args.tube_mode) == "pilot" else full


def _load_tube_targets(args: argparse.Namespace) -> pd.DataFrame:
    name = "tube_targets_pilot.parquet" if str(args.tube_mode) == "pilot" else "tube_targets_full.parquet"
    path = Path(args.out_dir) / "05_tube_target_generation" / name
    if not path.exists():
        return phase5_tube_targets(args)
    return pd.read_parquet(path)


def phase6_7_tube_inverse(args: argparse.Namespace) -> pd.DataFrame:
    out6 = Path(args.out_dir) / "06_tube_candidate_generation"
    out7 = Path(args.out_dir) / "07_tube_graph_branch_selection"
    out6.mkdir(parents=True, exist_ok=True)
    out7.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.robot_config))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    a1_pool = pd.read_parquet(args.a1_pool)
    mapping = infer_u_mapping_params(a1_pool)
    centerline = _load_centerline(args)
    tube_targets = _load_tube_targets(args)
    if int(args.max_tube_targets) > 0 and len(tube_targets) > int(args.max_tube_targets):
        tube_targets = tube_targets.head(int(args.max_tube_targets)).reset_index(drop=True)
    candidates, selected = _active_inverse_targets(
        tube_targets,
        a1_pool,
        args,
        inputs=inputs,
        theta_sign=theta_sign,
        mapping=mapping,
        centerline=centerline,
    )
    candidates.to_parquet(out6 / "tube_candidates.parquet", index=False, compression="zstd")
    pd.DataFrame([_quality_report(candidates)]).to_csv(out6 / "candidate_generation_report.csv", index=False)
    selected.to_parquet(out7 / "selected_tube_dataset.parquet", index=False, compression="zstd")
    report = {"mode": "greedy_active_inverse_selection", **_quality_report(selected)}
    (out7 / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    selected.groupby(["ellipse_id"], dropna=False)["xyz_residual_mm"].agg(["count", "mean", "max"]).to_csv(out7 / "per_angle_branch_smoothness.csv")
    return selected


def _load_tube_selected(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.out_dir) / "07_tube_graph_branch_selection" / "selected_tube_dataset.parquet"
    if not path.exists():
        return phase6_7_tube_inverse(args)
    return pd.read_parquet(path)


def phase8_assembly(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / "08_dataset_assembly"
    out_dir.mkdir(parents=True, exist_ok=True)
    center = _load_centerline(args)
    tube = _load_tube_selected(args)
    def parquet_ready(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].astype(str)
        return out

    parquet_ready(center).to_parquet(out_dir / "D_center.parquet", index=False, compression="zstd")
    parquet_ready(tube).to_parquet(out_dir / "D_tube_small.parquet", index=False, compression="zstd")
    parquet_ready(tube).to_parquet(out_dir / "D_tube_full.parquet", index=False, compression="zstd")
    a1 = pd.read_parquet(args.a1_pool)
    n_a1 = min(len(a1), max(1, int(round(0.3 / 0.7 * len(tube)))))
    mix = pd.concat([a1.sample(n=n_a1, random_state=int(args.seed)), tube], ignore_index=True)
    mix["sample_id"] = np.arange(len(mix), dtype=np.int64)
    parquet_ready(mix).to_parquet(out_dir / "D_mix_g30_t70.parquet", index=False, compression="zstd")
    report = pd.DataFrame(
        [
            {"dataset": "D_center", **_quality_report(center)},
            {"dataset": "D_tube_small", **_quality_report(tube)},
            {"dataset": "D_mix_g30_t70", **_quality_report(mix)},
        ]
    )
    write_markdown_table(report, out_dir / "dataset_quality_report.md", title="Dataset quality report")


def phase12_summary(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / "12_paper_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    support_path = Path(args.out_dir) / "02_support_recompute" / "support_summary_by_pool.csv"
    centerline_path = Path(args.out_dir) / "03_centerline_active_inverse" / "centerline_inverse_report.csv"
    dataset_report = Path(args.out_dir) / "08_dataset_assembly" / "dataset_quality_report.md"
    lines = [
        "# True sin_sin_cos tube canonical experiment summary",
        "",
        "旧 same-phase 结果不再作为真实闭合椭圆证据；本实验的 support 字段全部按当前 sin_sin_cos target 重新计算。",
        "",
        "## Key files",
        "",
        f"- support summary: `{support_path}`",
        f"- centerline inverse report: `{centerline_path}`",
        f"- dataset quality: `{dataset_report}`",
        f"- D_tube_small: `{Path(args.out_dir) / '08_dataset_assembly' / 'D_tube_small.parquet'}`",
        "",
    ]
    if support_path.exists():
        df = pd.read_csv(support_path)
        table_tmp = out_dir / "_support_summary_tmp.md"
        write_markdown_table(df, table_tmp, title="Support summary")
        support_md = "\n".join(table_tmp.read_text(encoding="utf-8").splitlines()[2:])
        table_tmp.unlink(missing_ok=True)
        lines += ["## Support summary", "", support_md, ""]
        df.to_csv(out_dir / "final_dataset_quality_table.csv", index=False)
    if centerline_path.exists():
        cdf = pd.read_csv(centerline_path)
        table_tmp = out_dir / "_centerline_tmp.md"
        write_markdown_table(cdf, table_tmp, title="Centerline inverse report")
        centerline_md = "\n".join(table_tmp.read_text(encoding="utf-8").splitlines()[2:])
        table_tmp.unlink(missing_ok=True)
        lines += [
            "## Centerline active inverse",
            "",
            centerline_md,
            "",
            "Gate: residual p95 must be <= 2mm for pilot. If this table is above that threshold, tube dataset generation and model training should not be treated as valid.",
            "",
        ]
    if dataset_report.exists():
        lines += ["## Dataset quality", "", dataset_report.read_text(encoding="utf-8"), ""]
    (out_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame().to_csv(out_dir / "final_model_quality_table.csv", index=False)
    (out_dir / "failure_diagnosis.md").write_text("See experiment_summary.md and per-phase reports.\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if _phase_selected(args, "phase0"):
        phase0_input_audit(args)
    if _phase_selected(args, "phase1"):
        phase1_generate_targets(args)
    if _phase_selected(args, "phase2"):
        phase2_support(args)
    if _phase_selected(args, "phase3"):
        phase3_centerline(args)
    if _phase_selected(args, "phase5"):
        phase5_tube_targets(args)
    if _phase_selected(args, "phase6") or _phase_selected(args, "phase7"):
        phase6_7_tube_inverse(args)
    if _phase_selected(args, "phase8"):
        phase8_assembly(args)
    if _phase_selected(args, "phase12"):
        phase12_summary(args)
    payload = {"out_dir": str(args.out_dir), "elapsed_s": time.perf_counter() - t0, "phases": args.phases}
    (Path(args.out_dir) / "run_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run true sin_sin_cos tube-aware canonical dataset experiment.")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "true_sinsincos_tube_canonical_v1")
    ap.add_argument("--phases", default="all")
    ap.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--a1-pool", type=Path, default=DEFAULT_A1_POOL)
    ap.add_argument("--hierarchical-pool", type=Path, default=DEFAULT_HIER_POOL)
    ap.add_argument("--enriched-pool", type=Path, default=DEFAULT_ENRICHED_POOL)
    ap.add_argument("--old-trajectory-points", type=Path, default=DEFAULT_13B_POINTS)
    ap.add_argument("--old-scoreboard", type=Path, default=DEFAULT_13B_SCOREBOARD)
    ap.add_argument("--support-pools", default="a1,old_enriched,hierarchical")
    ap.add_argument("--ellipses", default="E75,E87p5,E100")
    ap.add_argument("--n-points", type=int, default=360)
    ap.add_argument("--tube-mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--init-k", type=int, default=3)
    ap.add_argument("--inverse-max-nfev", type=int, default=80)
    ap.add_argument("--inverse-lambda-u", type=float, default=0.05)
    ap.add_argument("--inverse-residual-gate-mm", type=float, default=2.0)
    ap.add_argument("--max-angle-points", type=int, default=0)
    ap.add_argument("--max-tube-targets", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260709)
    return ap.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
