#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = REPO_ROOT / "scripts" / "baselines"
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import joblib
except Exception as exc:  # pragma: no cover - dependency failure is clearer at runtime.
    raise SystemExit(f"joblib is required to load the trained MLP: {exc}") from exc

from features import build_features  # noqa: E402
from fk_dh_numpy import fk_dh_batch  # noqa: E402
from run_baselines import XYScaler, _load_robot_paths, _read_lengths_end  # noqa: E402


DEFAULT_MODEL_DIR = (
    REPO_ROOT
    / "runs"
    / "mlp_capacity_fixed_layer_s1_0125_s2_0250_100k_relabel_v1"
    / "grid"
    / "relabel"
    / "iid"
    / "grid_relabel_iid_L2_alpha1em05_es"
    / "mlp_large"
)
DEFAULT_MODEL = DEFAULT_MODEL_DIR / "model.joblib"
DEFAULT_SCALER = DEFAULT_MODEL_DIR.parent / "scaler.joblib"
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "visualizations" / "best_mlp_circle_tracking_l2_relabel_iid"

DEFAULT_SAFE_RADIUS_BY_X = {
    1.00: 0.473,
    1.05: 0.429,
    1.10: 0.371,
    1.15: 0.292,
    1.20: 0.168,
}


class CircleCandidate(NamedTuple):
    candidate_id: str
    x0_m: float
    radius_m: float
    n_points: int


def _require_file(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")
    return path


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def make_circle_points(x0: float, radius: float, n_points: int, phase_rad: float = 0.0) -> np.ndarray:
    """Create a circle in the y-z plane at fixed x."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_points < 8:
        raise ValueError("n_points must be at least 8")
    angles = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False, dtype=float) + float(phase_rad)
    out = np.zeros((int(n_points), 3), dtype=float)
    out[:, 0] = float(x0)
    out[:, 1] = float(radius) * np.cos(angles)
    out[:, 2] = float(radius) * np.sin(angles)
    return out


def compute_tracking_metrics(
    target_xyz_m: np.ndarray,
    achieved_xyz_m: np.ndarray,
    tension_n: np.ndarray,
    tension_upper_n: float = 2000.0,
) -> dict[str, float]:
    target = np.asarray(target_xyz_m, dtype=float)
    achieved = np.asarray(achieved_xyz_m, dtype=float)
    tension = np.asarray(tension_n, dtype=float)
    if target.shape != achieved.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target_xyz_m and achieved_xyz_m must both have shape [N, 3]")
    if tension.ndim != 2:
        raise ValueError("tension_n must have shape [N, C]")

    err_mm = np.linalg.norm(achieved - target, axis=1) * 1000.0
    err_mm = _finite(err_mm)
    if err_mm.size == 0:
        raise ValueError("no finite tracking errors")

    tension_f = _finite(tension)
    if tension_f.size == 0:
        raise ValueError("no finite tensions")

    return {
        "ee_mean_mm": float(np.mean(err_mm)),
        "ee_rmse_mm": float(np.sqrt(np.mean(err_mm**2))),
        "ee_p50_mm": float(np.percentile(err_mm, 50)),
        "ee_p95_mm": float(np.percentile(err_mm, 95)),
        "ee_max_mm": float(np.max(err_mm)),
        "tension_min_n": float(np.min(tension_f)),
        "tension_max_n": float(np.max(tension_f)),
        "tension_mean_n": float(np.mean(tension_f)),
        "negative_tension_ratio": float(np.mean(tension_f < 0.0)),
        "over_upper_tension_ratio": float(np.mean(tension_f > float(tension_upper_n))),
    }


def select_best_candidate(
    rows: list[dict[str, float | str]],
    min_showcase_radius_m: float = 0.10,
    max_negative_tension_ratio: float = 0.0,
    max_over_upper_tension_ratio: float = 0.0,
) -> dict[str, float | str]:
    valid = []
    for row in rows:
        if float(row["radius_m"]) < float(min_showcase_radius_m):
            continue
        if float(row["negative_tension_ratio"]) > float(max_negative_tension_ratio):
            continue
        if float(row["over_upper_tension_ratio"]) > float(max_over_upper_tension_ratio):
            continue
        valid.append(row)
    if not valid:
        raise ValueError("no valid candidate circle after radius/tension filters")
    return min(valid, key=lambda r: (float(r["ee_p95_mm"]), float(r["ee_rmse_mm"]), -float(r["radius_m"])))


def build_candidate_grid(
    x_values: list[float],
    local_safe_radius_by_x: dict[float, float],
    fractions: list[float],
    min_radius_m: float,
    max_global_radius_m: float,
    n_points: int,
) -> list[CircleCandidate]:
    out: list[CircleCandidate] = []
    seen: set[tuple[float, float]] = set()
    for x0 in x_values:
        safe = float(local_safe_radius_by_x[round(float(x0), 2)])
        max_r = min(safe, float(max_global_radius_m))
        for frac in fractions:
            radius = round(max_r * float(frac), 6)
            if radius < float(min_radius_m) or radius > max_r:
                continue
            key = (round(float(x0), 6), radius)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                CircleCandidate(
                    candidate_id=f"x{float(x0):.2f}_r{radius:.3f}".replace(".", "p"),
                    x0_m=float(x0),
                    radius_m=radius,
                    n_points=int(n_points),
                )
            )
    return out


def circle_axis_limits(x0_m: float, radius_m: float, margin: float = 1.1) -> dict[str, tuple[float, float]]:
    half = float(radius_m) * float(margin)
    return {
        "x": (float(x0_m) - half, float(x0_m) + half),
        "y": (-half, half),
        "z": (-half, half),
    }


def _load_model_and_scaler(model_path: Path, scaler_path: Path):
    # Compatibility for scaler.joblib files written while run_baselines.py was executed as __main__.
    import __main__  # noqa: PLC0415

    if not hasattr(__main__, "XYScaler"):
        setattr(__main__, "XYScaler", XYScaler)
    return joblib.load(_require_file(model_path)), joblib.load(_require_file(scaler_path))


def predict_theta_tension(model, scaler: XYScaler, xyz_m: np.ndarray, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    features, _ = build_features(np.asarray(xyz_m, dtype=float), feature_set=feature_set)
    pred_norm = np.asarray(model.predict(scaler.transform_X(features)), dtype=float)
    if pred_norm.ndim != 2 or pred_norm.shape[1] < 2:
        raise ValueError(f"model prediction must be 2D with theta+tension columns, got {pred_norm.shape}")
    theta_dim = int(np.asarray(scaler.th_mean_).reshape(-1).shape[0])
    tension_dim = int(np.asarray(scaler.t_mean_).reshape(-1).shape[0])
    expected_dim = theta_dim + tension_dim
    if pred_norm.shape[1] != expected_dim:
        raise ValueError(f"model predicted {pred_norm.shape[1]} columns, expected {expected_dim}")
    theta, tension = scaler.inverse_Y(pred_norm[:, :theta_dim], pred_norm[:, theta_dim:])
    return np.asarray(theta, dtype=float), np.asarray(tension, dtype=float)


def evaluate_candidate(
    candidate: CircleCandidate,
    model,
    scaler: XYScaler,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    feature_set: str,
    tension_upper_n: float,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    target = make_circle_points(candidate.x0_m, candidate.radius_m, candidate.n_points)
    theta, tension = predict_theta_tension(model, scaler, target, feature_set)
    achieved = fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    metrics = compute_tracking_metrics(target, achieved, tension, tension_upper_n=tension_upper_n)

    row: dict[str, float | str] = {
        "candidate_id": candidate.candidate_id,
        "x0_m": float(candidate.x0_m),
        "radius_m": float(candidate.radius_m),
        "n_points": int(candidate.n_points),
        **metrics,
    }

    angles = np.linspace(0.0, 360.0, int(candidate.n_points), endpoint=False, dtype=float)
    err_mm = np.linalg.norm(achieved - target, axis=1) * 1000.0
    data = {
        "candidate_id": [candidate.candidate_id] * int(candidate.n_points),
        "angle_deg": angles,
        "target_x_m": target[:, 0],
        "target_y_m": target[:, 1],
        "target_z_m": target[:, 2],
        "achieved_x_m": achieved[:, 0],
        "achieved_y_m": achieved[:, 1],
        "achieved_z_m": achieved[:, 2],
        "ee_error_mm": err_mm,
    }
    for i in range(theta.shape[1]):
        data[f"theta{i + 1}_rad"] = theta[:, i]
    for i in range(tension.shape[1]):
        data[f"T{i + 1}_N"] = tension[:, i]
    return row, pd.DataFrame(data)


def plot_best_yz(points: pd.DataFrame, best: dict[str, float | str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.plot(points["target_y_m"], points["target_z_m"], color="#1f4e79", lw=2.2, label="target")
    ax.plot(points["achieved_y_m"], points["achieved_z_m"], color="#c7511f", lw=2.0, label="MLP + FK")
    ax.scatter(points["target_y_m"].iloc[0], points["target_z_m"].iloc[0], s=42, color="#1f4e79", marker="o")
    ax.scatter(points["achieved_y_m"].iloc[0], points["achieved_z_m"].iloc[0], s=42, color="#c7511f", marker="x")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"Best circle in y-z plane: x={float(best['x0_m']):.2f} m, r={float(best['radius_m']):.3f} m")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _savefig(fig, out_path)


def plot_best_3d(points: pd.DataFrame, best: dict[str, float | str], out_path: Path) -> None:
    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points["target_x_m"], points["target_y_m"], points["target_z_m"], color="#1f4e79", lw=2.2, label="target")
    ax.plot(points["achieved_x_m"], points["achieved_y_m"], points["achieved_z_m"], color="#c7511f", lw=2.0, label="MLP + FK")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"3D view: p95={float(best['ee_p95_mm']):.2f} mm, max={float(best['ee_max_mm']):.2f} mm")
    limits = circle_axis_limits(float(best["x0_m"]), float(best["radius_m"]), margin=1.15)
    ax.set_xlim(*limits["x"])
    ax.set_ylim(*limits["y"])
    ax.set_zlim(*limits["z"])
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-62)
    ax.legend(frameon=False)
    _savefig(fig, out_path)


def plot_error_profile(points: pd.DataFrame, best: dict[str, float | str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.plot(points["angle_deg"], points["ee_error_mm"], color="#6b5b95", lw=2.0)
    ax.axhline(float(best["ee_p95_mm"]), color="#c7511f", ls="--", lw=1.4, label="p95")
    ax.set_xlabel("circle angle (deg)")
    ax.set_ylabel("EE tracking error (mm)")
    ax.set_title("Open-loop tracking error along the selected circle")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _savefig(fig, out_path)


def plot_candidate_scan(summary: pd.DataFrame, best: dict[str, float | str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    sc = ax.scatter(
        summary["x0_m"],
        summary["radius_m"],
        c=summary["ee_p95_mm"],
        s=95,
        cmap="viridis_r",
        edgecolors="black",
        linewidths=0.4,
    )
    ax.scatter([best["x0_m"]], [best["radius_m"]], marker="*", s=260, color="#d62728", edgecolors="black", label="selected")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("EE p95 error (mm)")
    ax.set_xlabel("circle plane x (m)")
    ax.set_ylabel("circle radius (m)")
    ax.set_title("Candidate circle scan")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _savefig(fig, out_path)


def write_readme(out_dir: Path, best: dict[str, float | str], summary_path: Path, points_path: Path) -> None:
    text = f"""# Best MLP Circle Tracking

This folder visualizes the best open-loop circular trajectory found for the current fixed-layer relabel large MLP.

Pipeline:

1. Sample candidate circles in the plane perpendicular to the robot zero pose, i.e. fixed `x` and circular motion in `y-z`.
2. Run `target xyz -> MLP -> theta/T`.
3. Reconstruct `achieved xyz` with the project FK model.
4. Choose the display trajectory by lowest EE p95 error, after excluding very small radii and invalid tension candidates.

Selected candidate:

- `candidate_id`: `{best['candidate_id']}`
- `x`: `{float(best['x0_m']):.3f} m`
- `radius`: `{float(best['radius_m']):.3f} m`
- `points`: `{int(best['n_points'])}`
- `EE mean`: `{float(best['ee_mean_mm']):.3f} mm`
- `EE RMSE`: `{float(best['ee_rmse_mm']):.3f} mm`
- `EE p95`: `{float(best['ee_p95_mm']):.3f} mm`
- `EE max`: `{float(best['ee_max_mm']):.3f} mm`
- `tension min/max`: `{float(best['tension_min_n']):.2f} / {float(best['tension_max_n']):.2f} N`
- `negative tension ratio`: `{float(best['negative_tension_ratio']):.6f}`
- `over upper tension ratio`: `{float(best['over_upper_tension_ratio']):.6f}`

Files:

- `best_circle_tracking_yz.png`: target and achieved circle projected into the paper-style `y-z` plane.
- `best_circle_tracking_3d.png`: same trajectory in 3D coordinates.
- `best_circle_tracking_error_profile.png`: pointwise EE tracking error around the circle.
- `candidate_circle_scan.png`: all scanned `(x, radius)` candidates, colored by EE p95 error.
- `{summary_path.name}`: candidate-level metrics.
- `{points_path.name}`: point-level target, achieved, theta, tension, and error data for the selected candidate.

Note: this is an open-loop inverse-model visualization, not a closed-loop controller simulation.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, float | str]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, scaler = _load_model_and_scaler(Path(args.model), Path(args.scaler))
    lengths_csv, ee_csv = _load_robot_paths(args.robot_config, None, None)
    lengths_m, p_end_local_m = _read_lengths_end(lengths_csv, ee_csv)

    x_values = [float(v) for v in args.x_values.split(",") if v.strip()]
    fractions = [float(v) for v in args.radius_fractions.split(",") if v.strip()]
    candidates = build_candidate_grid(
        x_values=x_values,
        local_safe_radius_by_x=DEFAULT_SAFE_RADIUS_BY_X,
        fractions=fractions,
        min_radius_m=float(args.min_candidate_radius_m),
        max_global_radius_m=float(args.max_global_radius_m),
        n_points=int(args.n_points),
    )
    if not candidates:
        raise SystemExit("no candidates generated; relax x/radius filters")

    rows: list[dict[str, float | str]] = []
    frames: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        row, frame = evaluate_candidate(
            candidate=candidate,
            model=model,
            scaler=scaler,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            feature_set=args.feature_set,
            tension_upper_n=float(args.tension_upper_n),
        )
        rows.append(row)
        frames[candidate.candidate_id] = frame

    best = select_best_candidate(
        rows,
        min_showcase_radius_m=float(args.min_showcase_radius_m),
        max_negative_tension_ratio=float(args.max_negative_tension_ratio),
        max_over_upper_tension_ratio=float(args.max_over_upper_tension_ratio),
    )
    best_points = frames[str(best["candidate_id"])]
    summary = pd.DataFrame(rows).sort_values(["ee_p95_mm", "ee_rmse_mm", "radius_m"], ascending=[True, True, False])

    summary_path = out_dir / "candidate_circle_summary.csv"
    points_path = out_dir / "best_circle_tracking_points.parquet"
    json_path = out_dir / "best_circle_tracking_summary.json"
    summary.to_csv(summary_path, index=False)
    best_points.to_parquet(points_path, index=False)
    json_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_best_yz(best_points, best, out_dir / "best_circle_tracking_yz.png")
    plot_best_3d(best_points, best, out_dir / "best_circle_tracking_3d.png")
    plot_error_profile(best_points, best, out_dir / "best_circle_tracking_error_profile.png")
    plot_candidate_scan(summary, best, out_dir / "candidate_circle_scan.png")
    write_readme(out_dir, best, summary_path, points_path)
    return best


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Find and visualize the best paper-style circular trajectory for the large MLP.")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--scaler", default=str(DEFAULT_SCALER))
    ap.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--feature-set", default="poly_heavy", choices=["raw", "poly_medium", "poly_heavy"])
    ap.add_argument("--x-values", default="1.00,1.05,1.10,1.15,1.20")
    ap.add_argument("--radius-fractions", default="0.35,0.55,0.75,0.90")
    ap.add_argument("--n-points", type=int, default=240)
    ap.add_argument("--min-candidate-radius-m", type=float, default=0.05)
    ap.add_argument("--min-showcase-radius-m", type=float, default=0.10)
    ap.add_argument("--max-global-radius-m", type=float, default=0.38)
    ap.add_argument("--tension-upper-n", type=float, default=2000.0)
    ap.add_argument("--max-negative-tension-ratio", type=float, default=0.0)
    ap.add_argument("--max-over-upper-tension-ratio", type=float, default=0.0)
    return ap.parse_args()


def main() -> None:
    best = run(parse_args())
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
