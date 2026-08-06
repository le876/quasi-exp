#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
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


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
XYZ_COLS = ["x_m", "y_m", "z_m"]
DEFAULT_DATASETS = {
    "path_a": REPO_ROOT / "data" / "canonical_layer_field_u3_pilot_v1" / "path_a_sync_plus" / "jacobian_pass_pool.parquet",
    "path_d": REPO_ROOT / "data" / "canonical_layer_field_u3_pilot_v1" / "path_d_wide_redistribute" / "jacobian_pass_pool.parquet",
}


@dataclass(frozen=True)
class Candidate:
    dataset_id: str
    candidate_id: str
    center_x: float
    center_y: float
    center_z: float
    amp_xy_mm: float
    amp_z_mm: float
    phase_x: float
    phase_y: float
    phase_z: float
    orientation_id: str = "axis_phase"


def _parse_float_list(raw: str) -> list[float]:
    return [float(v.strip()) for v in str(raw).split(",") if v.strip()]


def _parse_deg_list(raw: str) -> list[float]:
    return [math.radians(float(v.strip())) for v in str(raw).split(",") if v.strip()]


def _candidate_id(c: Candidate) -> str:
    parts = [
        c.dataset_id,
        f"cx{c.center_x:.4f}",
        f"cy{c.center_y:.4f}",
        f"cz{c.center_z:.4f}",
        f"a{c.amp_xy_mm:.1f}",
        f"px{math.degrees(c.phase_x):.0f}",
        f"py{math.degrees(c.phase_y):.0f}",
        f"pz{math.degrees(c.phase_z):.0f}",
    ]
    return "_".join(parts).replace("-", "m").replace(".", "p")


def ellipse_points(c: Candidate, n_points: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False, dtype=float)
    amp_xy = float(c.amp_xy_mm) / 1000.0
    amp_z = float(c.amp_z_mm) / 1000.0
    xyz = np.empty((int(n_points), 3), dtype=float)
    xyz[:, 0] = float(c.center_x) + amp_xy * np.sin(t + float(c.phase_x))
    xyz[:, 1] = float(c.center_y) + amp_xy * np.sin(t + float(c.phase_y))
    xyz[:, 2] = float(c.center_z) + amp_z * np.sin(t + float(c.phase_z))
    return xyz


def has_fixed_axis_bias(values_mm: np.ndarray, threshold_mm: float = 2.0) -> bool:
    values = np.asarray(values_mm, dtype=float).reshape(-1)
    if len(values) == 0:
        return False
    return bool(np.all(values > float(threshold_mm)) or np.all(values < -float(threshold_mm)))


def load_dataset(path: Path, *, max_rows: int, seed: int) -> pd.DataFrame:
    cols = XYZ_COLS + BETA_COLS
    df = pd.read_parquet(path, columns=cols)
    if int(max_rows) > 0 and len(df) > int(max_rows):
        df = df.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)
    return df.reset_index(drop=True)


def choose_centers(df: pd.DataFrame, *, voxel_mm: float, max_centers: int, seed: int) -> pd.DataFrame:
    xyz = df[XYZ_COLS].to_numpy(dtype=float)
    voxel = np.floor(xyz / (float(voxel_mm) / 1000.0)).astype(np.int64)
    tmp = pd.DataFrame(voxel, columns=["vx", "vy", "vz"])
    tmp["row_id"] = np.arange(len(tmp), dtype=np.int64)
    counts = tmp.groupby(["vx", "vy", "vz"], sort=False)["row_id"].agg(["count", "first"]).reset_index()
    centers = df.iloc[counts["first"].to_numpy(dtype=np.int64)][XYZ_COLS].copy().reset_index(drop=True)
    centers["voxel_count"] = counts["count"].to_numpy(dtype=np.int64)
    centers["rho"] = np.sqrt(np.square(centers["y_m"].to_numpy(dtype=float)) + np.square(centers["z_m"].to_numpy(dtype=float)))
    centers["psi"] = np.arctan2(centers["z_m"].to_numpy(dtype=float), centers["y_m"].to_numpy(dtype=float))
    centers["x_bin"] = pd.cut(centers["x_m"], bins=min(12, max(2, len(centers) // 4)), labels=False, duplicates="drop")
    centers["rho_bin"] = pd.cut(centers["rho"], bins=min(8, max(2, len(centers) // 4)), labels=False, duplicates="drop")
    centers["psi_bin"] = pd.cut(centers["psi"], bins=12, labels=False, duplicates="drop")
    centers = centers.sort_values(["x_bin", "rho_bin", "psi_bin", "voxel_count"], ascending=[True, True, True, False])
    if len(centers) <= int(max_centers):
        return centers
    per_group = max(1, int(math.ceil(float(max_centers) / max(1, centers[["x_bin", "rho_bin", "psi_bin"]].drop_duplicates().shape[0]))))
    balanced = (
        centers.groupby(["x_bin", "rho_bin", "psi_bin"], dropna=False, sort=False)
        .head(per_group)
        .sort_values("voxel_count", ascending=False)
    )
    if len(balanced) > int(max_centers):
        balanced = balanced.sample(n=int(max_centers), random_state=int(seed), weights=np.maximum(balanced["voxel_count"], 1))
    return balanced.reset_index(drop=True)


def build_candidates(
    dataset_id: str,
    centers: pd.DataFrame,
    *,
    amp_xy_mm: list[float],
    phases_rad: list[float],
    phase_mode: str,
) -> list[Candidate]:
    out: list[Candidate] = []
    if phase_mode == "diagonal":
        phase_tuples = [(p, p, p + math.pi / 2.0) for p in phases_rad]
    elif phase_mode == "xy_locked":
        phase_tuples = [(pxy, pxy, pz) for pxy in phases_rad for pz in phases_rad]
    elif phase_mode == "full":
        phase_tuples = [(px, py, pz) for px in phases_rad for py in phases_rad for pz in phases_rad]
    else:
        raise ValueError(f"unsupported phase_mode: {phase_mode}")
    for row in centers.itertuples(index=False):
        for amp in amp_xy_mm:
            for px, py, pz in phase_tuples:
                base = Candidate(
                    dataset_id=dataset_id,
                    candidate_id="",
                    center_x=float(row.x_m),
                    center_y=float(row.y_m),
                    center_z=float(row.z_m),
                    amp_xy_mm=float(amp),
                    amp_z_mm=1.5 * float(amp),
                    phase_x=float(px),
                    phase_y=float(py),
                    phase_z=float(pz),
                )
                out.append(Candidate(**{**asdict(base), "candidate_id": _candidate_id(base)}))
    return out


def support_metrics(c: Candidate, xyz: np.ndarray, nn: NearestNeighbors, *, n_points: int, max_nn_p95_mm: float, max_x_bias_mm: float) -> tuple[dict[str, Any], np.ndarray]:
    target = ellipse_points(c, n_points)
    dist, idx = nn.kneighbors(target, n_neighbors=1)
    nearest = xyz[idx[:, 0]]
    dist_mm = dist[:, 0] * 1000.0
    x_diff_mm = (nearest[:, 0] - target[:, 0]) * 1000.0
    row = asdict(c)
    row.update(
        {
            "nn_mean_mm": float(np.mean(dist_mm)),
            "nn_p95_mm": float(np.percentile(dist_mm, 95)),
            "nn_max_mm": float(np.max(dist_mm)),
            "nearest_x_mean_diff_mm": float(np.mean(x_diff_mm)),
            "nearest_x_p95_abs_diff_mm": float(np.percentile(np.abs(x_diff_mm), 95)),
            "fixed_nearest_x_bias_gt2mm": has_fixed_axis_bias(x_diff_mm, threshold_mm=max_x_bias_mm),
            "support_gate_pass": bool(np.percentile(dist_mm, 95) <= float(max_nn_p95_mm) and abs(float(np.mean(x_diff_mm))) <= float(max_x_bias_mm)),
            "tube_count_p10": float("nan"),
            "tube_normal_thickness_2_mm": float("nan"),
            "tube_normal_thickness_3_mm": float("nan"),
            "tube_beta_rms_p95_deg": float("nan"),
            "branch_gate_pass": False,
            "detailed_evaluated": False,
        }
    )
    return row, idx[:, 0].astype(np.int64)


def detailed_tube_metrics(
    c: Candidate,
    df: pd.DataFrame,
    radius_nn: NearestNeighbors,
    xyz: np.ndarray,
    *,
    n_points: int,
    tube_radius_mm: float,
    beta_pair_radius_mm: float,
    beta_gate_deg: float,
    min_count: int,
) -> dict[str, Any]:
    target = ellipse_points(c, n_points)
    radius_m = float(tube_radius_mm) / 1000.0
    ind = radius_nn.radius_neighbors(target, radius=radius_m, return_distance=False)
    counts = np.asarray([len(v) for v in ind], dtype=float)
    unique_idx = np.unique(np.concatenate([v for v in ind if len(v) > 0])) if np.any(counts > 0) else np.empty((0,), dtype=np.int64)
    thickness2 = float("nan")
    thickness3 = float("nan")
    if len(unique_idx) >= 6:
        tube_xyz = xyz[unique_idx]
        cov = np.cov((tube_xyz - tube_xyz.mean(axis=0)).T)
        vals = np.linalg.eigvalsh(cov)
        vals = np.sort(np.maximum(vals, 0.0))[::-1]
        thickness2 = float(math.sqrt(vals[1]) * 1000.0)
        thickness3 = float(math.sqrt(vals[2]) * 1000.0)

    beta_p95 = float("inf")
    if len(unique_idx) >= 2:
        local_xyz = xyz[unique_idx]
        local_beta = df.iloc[unique_idx][BETA_COLS].to_numpy(dtype=float)
        k = min(16, len(unique_idx))
        nn_local = NearestNeighbors(n_neighbors=k).fit(local_xyz)
        dist, idx = nn_local.kneighbors(local_xyz)
        vals: list[float] = []
        pair_radius_m = float(beta_pair_radius_mm) / 1000.0
        for i in range(len(local_xyz)):
            mask = (dist[i] > 1.0e-12) & (dist[i] <= pair_radius_m)
            if not np.any(mask):
                continue
            diff = local_beta[idx[i][mask]] - local_beta[i]
            rms = np.sqrt(np.mean(np.square(diff), axis=1)) * (180.0 / math.pi)
            vals.extend(float(v) for v in rms)
        if vals:
            beta_p95 = float(np.percentile(vals, 95))
        else:
            beta_p95 = float("nan")

    return {
        "tube_count_p10": float(np.percentile(counts, 10)) if len(counts) else 0.0,
        "tube_count_median": float(np.percentile(counts, 50)) if len(counts) else 0.0,
        "tube_unique_rows": int(len(unique_idx)),
        "tube_normal_thickness_2_mm": thickness2,
        "tube_normal_thickness_3_mm": thickness3,
        "tube_beta_rms_p95_deg": beta_p95,
        "branch_gate_pass": bool(
            np.isfinite(beta_p95)
            and beta_p95 <= float(beta_gate_deg)
            and np.percentile(counts, 10) >= float(min_count)
            and np.isfinite(thickness2)
            and np.isfinite(thickness3)
            and thickness2 >= 5.0
            and thickness3 >= 3.0
        ),
        "detailed_evaluated": True,
    }


def evaluate_dataset(dataset_id: str, path: Path, args: argparse.Namespace) -> pd.DataFrame:
    df = load_dataset(path, max_rows=int(args.max_rows), seed=int(args.seed))
    xyz = df[XYZ_COLS].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=1).fit(xyz)
    radius_nn = NearestNeighbors(algorithm="auto").fit(xyz)
    centers = choose_centers(df, voxel_mm=float(args.voxel_mm), max_centers=int(args.max_centers), seed=int(args.seed))
    candidates = build_candidates(
        dataset_id,
        centers,
        amp_xy_mm=_parse_float_list(args.amp_xy_mm),
        phases_rad=_parse_deg_list(args.phase_deg),
        phase_mode=str(args.phase_mode),
    )
    if int(args.max_candidates) > 0 and len(candidates) > int(args.max_candidates):
        rng = np.random.default_rng(int(args.seed))
        keep = rng.choice(len(candidates), size=int(args.max_candidates), replace=False)
        candidates = [candidates[i] for i in np.sort(keep)]

    rows: list[dict[str, Any]] = []
    for i, candidate in enumerate(candidates):
        row, _nearest = support_metrics(
            candidate,
            xyz,
            nn,
            n_points=int(args.n_points),
            max_nn_p95_mm=float(args.max_nn_p95_mm),
            max_x_bias_mm=float(args.max_x_bias_mm),
        )
        rows.append(row)
        if bool(args.progress) and (i + 1) % 1000 == 0:
            print(f"{dataset_id}: evaluated {i + 1}/{len(candidates)} support candidates", flush=True)

    out = pd.DataFrame(rows)
    if not out.empty:
        detailed_idx: set[int] = set()
        for amp, part in out.groupby("amp_xy_mm", sort=False):
            ranked = part.sort_values(["support_gate_pass", "nn_p95_mm", "nn_max_mm"], ascending=[False, True, True])
            detailed_idx.update(int(i) for i in ranked.head(int(args.max_detailed_per_amp)).index.tolist())
        for idx in sorted(detailed_idx):
            c = candidates[idx]
            detail = detailed_tube_metrics(
                c,
                df,
                radius_nn,
                xyz,
                n_points=int(args.n_points),
                tube_radius_mm=float(args.tube_radius_mm),
                beta_pair_radius_mm=float(args.beta_pair_radius_mm),
                beta_gate_deg=float(args.beta_gate_deg),
                min_count=int(args.min_tube_count_p10),
            )
            for key, value in detail.items():
                out.loc[idx, key] = value
    return out


def plot_top(out_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(7 * max(1, len(frames)), 5), squeeze=False)
    for ax, (dataset_id, df) in zip(axes[0], frames.items()):
        if df.empty:
            ax.set_title(dataset_id)
            continue
        plot_df = df[df["detailed_evaluated"].astype(bool)].copy()
        if plot_df.empty:
            plot_df = df.copy()
        scatter = ax.scatter(
            plot_df["amp_xy_mm"],
            plot_df["nn_p95_mm"],
            c=plot_df["nearest_x_mean_diff_mm"],
            cmap="coolwarm",
            s=20,
            alpha=0.8,
        )
        ax.axhline(8.0, color="#444", lw=1, ls="--")
        ax.axhline(5.0, color="#111", lw=1, ls=":")
        ax.set_title(dataset_id)
        ax.set_xlabel("amp_xy (mm)")
        ax.set_ylabel("NN p95 (mm)")
        fig.colorbar(scatter, ax=ax, label="nearest x mean diff (mm)")
    fig.tight_layout()
    fig.savefig(out_dir / "top_supported_ellipses.png", dpi=180)
    plt.close(fig)


def write_top_md(out_dir: Path, frames: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    lines = [
        "# Existing pool free-phase ellipse search",
        "",
        f"- n_points: `{args.n_points}`",
        f"- max_centers: `{args.max_centers}`",
        f"- phase_mode: `{args.phase_mode}`",
        f"- support gate: `nn_p95 <= {args.max_nn_p95_mm} mm` and `abs(x_bias) <= {args.max_x_bias_mm} mm`",
        f"- branch gate: `tube beta p95 <= {args.beta_gate_deg} deg`, tube count/thickness gates",
        "",
    ]
    cols = [
        "dataset_id",
        "candidate_id",
        "amp_xy_mm",
        "amp_z_mm",
        "center_x",
        "center_y",
        "center_z",
        "nn_p95_mm",
        "nearest_x_mean_diff_mm",
        "tube_count_p10",
        "tube_normal_thickness_2_mm",
        "tube_normal_thickness_3_mm",
        "tube_beta_rms_p95_deg",
        "support_gate_pass",
        "branch_gate_pass",
    ]
    for dataset_id, df in frames.items():
        lines += [f"## {dataset_id}", ""]
        if df.empty:
            lines += ["No candidates.", ""]
            continue
        detailed = df[df["detailed_evaluated"].astype(bool)].copy()
        qualified = detailed[(detailed["support_gate_pass"].astype(bool)) & (detailed["branch_gate_pass"].astype(bool))]
        if qualified.empty:
            best = detailed.sort_values(["support_gate_pass", "amp_xy_mm", "nn_p95_mm"], ascending=[False, False, True]).head(10)
            lines.append("No candidate passed both support and branch gates. Best detailed candidates:")
        else:
            best = qualified.sort_values(["amp_xy_mm", "nn_p95_mm"], ascending=[False, True]).head(10)
            lines.append("Top candidates passing both support and branch gates:")
        lines.append("")
        lines.append(simple_markdown_table(best[[c for c in cols if c in best.columns]]))
        lines.append("")
    (out_dir / "top_supported_ellipses.md").write_text("\n".join(lines), encoding="utf-8")


def simple_markdown_table(df: pd.DataFrame, *, max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    shown = df.head(int(max_rows)).copy()
    headers = [str(c) for c in shown.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in shown.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                if math.isfinite(value):
                    vals.append(f"{value:.4g}")
                else:
                    vals.append("")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for item in str(args.datasets).split(","):
        key = item.strip()
        if not key:
            continue
        if key not in DEFAULT_DATASETS:
            raise ValueError(f"unknown dataset {key}; available={sorted(DEFAULT_DATASETS)}")
        datasets[key] = DEFAULT_DATASETS[key]

    frames = {}
    for dataset_id, path in datasets.items():
        frame = evaluate_dataset(dataset_id, path, args)
        frame.to_csv(out_dir / f"{dataset_id}_candidates.csv", index=False)
        frames[dataset_id] = frame
    write_top_md(out_dir, frames, args)
    plot_top(out_dir, frames)
    summary = {
        "mode": "existing_pool_free_phase_ellipse_search",
        "out_dir": str(out_dir),
        "datasets": {
            dataset_id: {
                "rows": int(len(df)),
                "detailed_rows": int(df["detailed_evaluated"].sum()) if not df.empty else 0,
                "support_pass": int(df["support_gate_pass"].sum()) if not df.empty else 0,
                "branch_pass": int(df["branch_gate_pass"].sum()) if not df.empty else 0,
                "max_branch_pass_amp_xy_mm": float(df.loc[df["branch_gate_pass"].astype(bool), "amp_xy_mm"].max())
                if (not df.empty and df["branch_gate_pass"].astype(bool).any())
                else None,
            }
            for dataset_id, df in frames.items()
        },
    }
    (out_dir / "search_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search free-phase 1:1:1.5 ellipses supported by existing canonical pools.")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "01_existing_pool_free_ellipse_search")
    ap.add_argument("--datasets", default="path_a,path_d")
    ap.add_argument("--amp-xy-mm", default="25,37.5,50,62.5,75,87.5,100")
    ap.add_argument("--phase-deg", default="0,90,180,270")
    ap.add_argument("--phase-mode", choices=["diagonal", "xy_locked", "full"], default="xy_locked")
    ap.add_argument("--n-points", type=int, default=360)
    ap.add_argument("--voxel-mm", type=float, default=10.0)
    ap.add_argument("--max-centers", type=int, default=500)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-detailed-per-amp", type=int, default=20)
    ap.add_argument("--max-nn-p95-mm", type=float, default=8.0)
    ap.add_argument("--max-x-bias-mm", type=float, default=2.0)
    ap.add_argument("--tube-radius-mm", type=float, default=15.0)
    ap.add_argument("--beta-pair-radius-mm", type=float, default=10.0)
    ap.add_argument("--beta-gate-deg", type=float, default=2.0)
    ap.add_argument("--min-tube-count-p10", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--progress", action="store_true")
    return ap.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
