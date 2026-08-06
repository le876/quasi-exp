#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from features import build_features  # noqa: E402
from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from true_sinsincos_tube_utils import beta_from_u_deg, recompute_support_for_targets  # noqa: E402


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]


def theta_from_beta_batch(beta: np.ndarray, *, theta_sign: float) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).reshape(-1, 6)
    theta = np.empty((len(beta), 30), dtype=float)
    for section, (odd_col, even_col) in enumerate(((0, 1), (2, 3), (4, 5))):
        start = section * 10
        theta[:, start : start + 10 : 2] = beta[:, odd_col : odd_col + 1]
        theta[:, start + 1 : start + 10 : 2] = beta[:, even_col : even_col + 1]
    return theta * float(theta_sign)


def ellipse_points(row: pd.Series, n_points: int, *, trajectory_family: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False)
    phase_x = float(row["phase_x"])
    phase_y = float(row["phase_y"])
    phase_z = float(row["phase_z"])
    if trajectory_family == "same_phase":
        pass
    elif trajectory_family == "sin_sin_cos":
        phase_y = phase_x
        phase_z = phase_x + math.pi / 2.0
    else:
        raise ValueError(f"unsupported trajectory family: {trajectory_family}")
    xyz = np.empty((len(t), 3), dtype=float)
    xyz[:, 0] = float(row["center_x"]) + float(row["amp_xy_mm"]) / 1000.0 * np.sin(t + phase_x)
    xyz[:, 1] = float(row["center_y"]) + float(row["amp_xy_mm"]) / 1000.0 * np.sin(t + phase_y)
    xyz[:, 2] = float(row["center_z"]) + float(row["amp_z_mm"]) / 1000.0 * np.sin(t + phase_z)
    return xyz, t


def axis_metrics(target_xyz: np.ndarray, achieved_xyz: np.ndarray) -> dict[str, Any]:
    err = (achieved_xyz - target_xyz) * 1000.0
    out: dict[str, Any] = {}
    fixed = []
    p95s = []
    for i, axis in enumerate("xyz"):
        vals = err[:, i]
        out[f"{axis}err_mean_mm"] = float(np.mean(vals))
        out[f"{axis}err_p95_abs_mm"] = float(np.percentile(np.abs(vals), 95))
        out[f"{axis}err_same_sign_ratio"] = float(max(np.mean(vals > 0.0), np.mean(vals < 0.0)))
        out[f"fixed_{axis}_bias_gt2mm"] = bool(np.all(vals > 2.0) or np.all(vals < -2.0))
        fixed.append(out[f"fixed_{axis}_bias_gt2mm"])
        p95s.append(out[f"{axis}err_p95_abs_mm"])
    out["axiserr_max_p95_abs_mm"] = float(max(p95s))
    out["fixed_any_axis_bias_gt2mm"] = bool(any(fixed))
    return out


def load_model_specs(model_root: Path) -> list[tuple[str, Path]]:
    paths = sorted(model_root.glob("*/model.joblib"))
    return [(p.parent.name, p) for p in paths]


def predict_pose(package: dict[str, Any], target_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features, _names = build_features(target_xyz, feature_set=str(package["feature_set"]))
    pred = package["y_scaler"].inverse_transform(package["model"].predict(package["x_scaler"].transform(features)))
    if package["kind"] == "beta6_pose":
        beta = pred
        theta = theta_from_beta_batch(beta, theta_sign=float(package.get("theta_sign", -1.0)))
    elif package["kind"] == "theta30_direct_pose":
        theta = pred
        beta = np.full((len(theta), 6), np.nan, dtype=float)
    elif package["kind"] == "u_pose":
        mapping = dict(package.get("u_mapping", {}))
        missing = [k for k in ["s10", "s20", "ds1", "ds2", "dq"] if k not in mapping]
        if missing:
            raise ValueError(f"u_pose package missing u_mapping keys: {missing}")
        beta, _s1, _s2, _q = beta_from_u_deg(pred, **mapping)
        theta = theta_from_beta_batch(beta, theta_sign=float(package.get("theta_sign", -1.0)))
    else:
        raise ValueError(f"unsupported model package kind: {package.get('kind')}")
    return beta, theta


def evaluate_candidate(
    model_id: str,
    package: dict[str, Any],
    cand: pd.Series,
    *,
    n_points: int,
    trajectory_family: str,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    target, t = ellipse_points(cand, n_points, trajectory_family=trajectory_family)
    beta, theta = predict_pose(package, target)
    achieved = fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    ee = np.linalg.norm(achieved - target, axis=1) * 1000.0
    row: dict[str, Any] = {
        "model_id": model_id,
        "model_kind": package["kind"],
        "trajectory_family": trajectory_family,
        "candidate_id": cand["candidate_id"],
        "amp_xy_mm": float(cand["amp_xy_mm"]),
        "amp_z_mm": float(cand["amp_z_mm"]),
        "center_x": float(cand["center_x"]),
        "center_y": float(cand["center_y"]),
        "center_z": float(cand["center_z"]),
        "inherited_nn_p95_mm": float(cand.get("nn_p95_mm", np.nan)),
        "inherited_nearest_x_mean_diff_mm": float(cand.get("nearest_x_mean_diff_mm", np.nan)),
        "inherited_tube_count_p10": float(cand.get("tube_count_p10", np.nan)),
        "inherited_tube_beta_rms_p95_deg": float(cand.get("tube_beta_rms_p95_deg", np.nan)),
        "true_nn_p95_mm": float(cand.get("true_nn_p95_mm", np.nan)),
        "true_tube_count_p10": float(cand.get("true_tube_count_p10", np.nan)),
        "true_tube_beta_rms_p95_deg": float(cand.get("true_tube_beta_rms_p95_deg", np.nan)),
        "true_support_gate_pass": bool(cand.get("true_support_gate_pass", False)),
        "ee_mean_mm": float(np.mean(ee)),
        "ee_p95_mm": float(np.percentile(ee, 95)),
        "ee_max_mm": float(np.max(ee)),
    }
    row.update(axis_metrics(target, achieved))
    points = pd.DataFrame(
        {
            "model_id": model_id,
            "trajectory_family": trajectory_family,
            "candidate_id": cand["candidate_id"],
            "angle_rad": t,
            "target_x_m": target[:, 0],
            "target_y_m": target[:, 1],
            "target_z_m": target[:, 2],
            "achieved_x_m": achieved[:, 0],
            "achieved_y_m": achieved[:, 1],
            "achieved_z_m": achieved[:, 2],
            "ee_err_mm": ee,
        }
    )
    for i, col in enumerate(BETA_COLS):
        points[f"pred_{col}"] = beta[:, i]
    return row, points


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def select_candidates(path: Path, *, max_candidates: int, require_true_support: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "detailed_evaluated" in df.columns:
        df = df[df["detailed_evaluated"].astype(bool)].copy()
    if bool(require_true_support):
        if "true_support_gate_pass" not in df.columns:
            raise SystemExit("require_true_support=True but candidates do not contain true_support_gate_pass")
        df = df[_as_bool(df["true_support_gate_pass"])].copy()
        sort_cols = ["amp_xy_mm", "true_nn_p95_mm"]
        ascending = [False, True]
        if "true_nearest_x_mean_diff_mm" in df.columns:
            df["_true_x_abs"] = np.abs(df["true_nearest_x_mean_diff_mm"].to_numpy(dtype=float))
            sort_cols.append("_true_x_abs")
            ascending.append(True)
    elif "support_gate_pass" in df.columns:
        df = df[_as_bool(df["support_gate_pass"])].copy()
    if df.empty:
        raise SystemExit("no supported detailed candidates found")
    if not bool(require_true_support):
        sort_cols = ["amp_xy_mm", "nn_p95_mm", "nearest_x_mean_diff_mm"]
        ascending = [False, True, True]
    return df.sort_values(sort_cols, ascending=ascending).head(int(max_candidates)).drop(columns=["_true_x_abs"], errors="ignore").reset_index(drop=True)


def build_candidate_targets(candidates: pd.DataFrame, *, n_points: int, trajectory_family: str) -> pd.DataFrame:
    frames = []
    for _idx, cand in candidates.iterrows():
        target, angle = ellipse_points(cand, int(n_points), trajectory_family=trajectory_family)
        frame = pd.DataFrame(
            {
                "candidate_id": cand["candidate_id"],
                "ellipse_id": cand.get("ellipse_id", cand["candidate_id"]),
                "angle_idx": np.arange(len(target), dtype=np.int64),
                "angle_rad": angle,
                "x_target_m": target[:, 0],
                "y_target_m": target[:, 1],
                "z_target_m": target[:, 2],
                "amp_xy_mm": float(cand["amp_xy_mm"]),
                "amp_z_mm": float(cand["amp_z_mm"]),
                "center_x": float(cand["center_x"]),
                "center_y": float(cand["center_y"]),
                "center_z": float(cand["center_z"]),
                "trajectory_family": trajectory_family,
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_true_support(
    candidates: pd.DataFrame,
    support_pool: pd.DataFrame,
    *,
    n_points: int,
    trajectory_family: str,
    pool_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = build_candidate_targets(candidates, n_points=n_points, trajectory_family=trajectory_family)
    summary, per_angle = recompute_support_for_targets(targets, support_pool, pool_label=pool_label)
    drop_cols = [c for c in summary.columns if c in candidates.columns and c not in {"candidate_id"}]
    merged = candidates.merge(summary.drop(columns=drop_cols), on="candidate_id", how="left", validate="one_to_one")
    return merged, per_angle


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def set_equal_3d_limits(ax: Any, points: pd.DataFrame) -> None:
    cols = [
        "target_x_m",
        "target_y_m",
        "target_z_m",
        "achieved_x_m",
        "achieved_y_m",
        "achieved_z_m",
    ]
    xyz = points[cols].to_numpy(dtype=float).reshape(-1, 3)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if len(xyz) == 0:
        return
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.max(hi - lo) / 2.0)
    radius = max(radius * 1.08, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def plot_3d(points: pd.DataFrame, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points["target_x_m"], points["target_y_m"], points["target_z_m"], color="#1f4e79", lw=2.0, label="target")
    ax.plot(points["achieved_x_m"], points["achieved_y_m"], points["achieved_z_m"], color="#c4511c", lw=1.7, label="model + FK")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    set_equal_3d_limits(ax, points)
    ax.view_init(elev=24, azim=-42)
    ax.set_title(title)
    ax.legend(loc="upper right")
    _savefig(fig, out_path)


def plot_axis_errors(points: pd.DataFrame, title: str, out_path: Path) -> None:
    angle_deg = points["angle_rad"].to_numpy(dtype=float) * 180.0 / math.pi
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for axis in "xyz":
        vals = (points[f"achieved_{axis}_m"] - points[f"target_{axis}_m"]) * 1000.0
        ax.plot(angle_deg, vals, lw=1.3, label=f"{axis} err")
    ax.axhline(0.0, color="#333", lw=0.8)
    ax.set_xlabel("ellipse angle (deg)")
    ax.set_ylabel("axis error (mm)")
    ax.set_title(title)
    ax.legend(loc="best")
    _savefig(fig, out_path)


def plot_grid(points: pd.DataFrame, scoreboard: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for model_id, part in scoreboard.groupby("model_id"):
        best = part.sort_values(["ee_p95_mm", "axiserr_max_p95_abs_mm", "amp_xy_mm"], ascending=[True, True, False]).iloc[0]
        rows.append(best)
    rows = rows[:9]
    n = len(rows)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(5 * ncols, 4.5 * nrows))
    for i, row in enumerate(rows, start=1):
        ax = fig.add_subplot(nrows, ncols, i, projection="3d")
        part = points[(points["model_id"] == row["model_id"]) & (points["candidate_id"] == row["candidate_id"])]
        ax.plot(part["target_x_m"], part["target_y_m"], part["target_z_m"], color="#1f4e79", lw=1.7)
        ax.plot(part["achieved_x_m"], part["achieved_y_m"], part["achieved_z_m"], color="#c4511c", lw=1.3)
        set_equal_3d_limits(ax, part)
        ax.view_init(elev=24, azim=-42)
        ax.set_title(f"{row['model_id']}\nEE95={row['ee_p95_mm']:.2f}mm a={row['amp_xy_mm']:.1f}mm", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.tight_layout()
    _savefig(fig, out_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.robot_config))
    inputs = load_robot_inputs(cfg)
    raw_candidates = pd.read_csv(Path(args.candidates_csv))
    if getattr(args, "support_pool", None) is not None and bool(args.recompute_support):
        support_pool = pd.read_parquet(args.support_pool)
        raw_candidates, per_angle_support = attach_true_support(
            raw_candidates,
            support_pool,
            n_points=int(args.n_points),
            trajectory_family=str(args.trajectory_family),
            pool_label=Path(args.support_pool).stem,
        )
        raw_candidates.to_csv(out_dir / "candidates_with_true_support.csv", index=False)
        per_angle_support.to_parquet(out_dir / "true_per_angle_support.parquet", index=False, compression="zstd")
        candidates_csv = out_dir / "candidates_with_true_support.csv"
    else:
        candidates_csv = Path(args.candidates_csv)
    candidates = select_candidates(
        candidates_csv,
        max_candidates=int(args.max_candidates),
        require_true_support=bool(getattr(args, "require_true_support", False)),
    )
    models = load_model_specs(Path(args.model_root))
    rows = []
    point_frames = []
    for model_id, model_path in models:
        package = load(model_path)
        for _idx, cand in candidates.iterrows():
            row, points = evaluate_candidate(
                model_id,
                package,
                cand,
                n_points=int(args.n_points),
                trajectory_family=str(args.trajectory_family),
                lengths_m=inputs.lengths_m,
                p_end_local_m=inputs.p_end_local_m,
            )
            rows.append(row)
            point_frames.append(points)
    scoreboard = pd.DataFrame(rows).sort_values(["ee_p95_mm", "axiserr_max_p95_abs_mm", "amp_xy_mm"], ascending=[True, True, False])
    all_points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    scoreboard.to_csv(out_dir / "global_showcase_scoreboard.csv", index=False)
    all_points.to_parquet(out_dir / "trajectory_points.parquet", index=False, compression="zstd")

    qualified = scoreboard[
        (scoreboard["ee_p95_mm"] <= float(args.quality_ee_p95_mm))
        & (scoreboard["axiserr_max_p95_abs_mm"] <= float(args.quality_axis_p95_mm))
        & (~scoreboard["fixed_any_axis_bias_gt2mm"].astype(bool))
    ].copy()
    if qualified.empty:
        best = scoreboard.iloc[0]
    else:
        best = qualified.sort_values(["amp_xy_mm", "ee_p95_mm"], ascending=[False, True]).iloc[0]
    best_points = all_points[(all_points["model_id"] == best["model_id"]) & (all_points["candidate_id"] == best["candidate_id"])]
    plot_3d(best_points, f"best: {best['model_id']} / a={best['amp_xy_mm']:.1f}mm / EE95={best['ee_p95_mm']:.2f}mm", out_dir / "best_largest_qualified_3d.png")
    plot_axis_errors(best_points, f"axis errors: {best['model_id']} / {best['candidate_id']}", out_dir / "best_largest_qualified_axis_errors.png")
    plot_grid(all_points, scoreboard, out_dir / "all_models_largest_qualified_grid.png")

    payload = {
        "mode": "beta6_large_ellipse_benchmark",
        "trajectory_family": str(args.trajectory_family),
        "best": best.to_dict(),
        "models": [m[0] for m in models],
        "candidates": int(len(candidates)),
    }
    (out_dir / "benchmark_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark beta6 pose models on large supported ellipses.")
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--candidates-csv", type=Path, default=REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "01_existing_pool_free_ellipse_search" / "path_a_candidates.csv")
    ap.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=360)
    ap.add_argument("--support-pool", type=Path, default=None)
    ap.add_argument("--recompute-support", action="store_true")
    ap.add_argument("--require-true-support", action="store_true")
    ap.add_argument(
        "--trajectory-family",
        choices=["same_phase", "sin_sin_cos"],
        default="same_phase",
        help="same_phase keeps legacy px=py=pz paths; sin_sin_cos uses x/y=sin and z=cos.",
    )
    ap.add_argument("--quality-ee-p95-mm", type=float, default=5.0)
    ap.add_argument("--quality-axis-p95-mm", type=float, default=3.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    payload = run(args)
    print(json.dumps({"out_dir": str(args.out_dir), "best": payload["best"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
