#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
XYZ_COLS = ["x_m", "y_m", "z_m"]
DEFAULT_DATASETS = {
    "path_a": REPO_ROOT / "data" / "canonical_layer_field_u3_pilot_v1" / "path_a_sync_plus" / "jacobian_pass_pool.parquet",
    "path_d": REPO_ROOT / "data" / "canonical_layer_field_u3_pilot_v1" / "path_d_wide_redistribute" / "jacobian_pass_pool.parquet",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _bin_index(values: np.ndarray, lo: float, step: float, n_bins: int) -> np.ndarray:
    idx = np.floor((np.asarray(values, dtype=float) - float(lo)) / float(step)).astype(np.int64)
    return np.clip(idx, 0, int(n_bins) - 1)


def shell_frame(df: pd.DataFrame, *, x_bin_mm: float, rho_bin_mm: float, psi_bins: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    xyz = df[XYZ_COLS].to_numpy(dtype=float)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    rho = np.sqrt(y * y + z * z)
    psi = np.arctan2(z, y)

    x_step = float(x_bin_mm) / 1000.0
    rho_step = float(rho_bin_mm) / 1000.0
    x_min = math.floor(float(np.min(x)) / x_step) * x_step
    x_max = math.ceil(float(np.max(x)) / x_step) * x_step
    rho_min = math.floor(float(np.min(rho)) / rho_step) * rho_step
    rho_max = math.ceil(float(np.max(rho)) / rho_step) * rho_step
    n_x = max(1, int(math.ceil((x_max - x_min) / x_step)))
    n_rho = max(1, int(math.ceil((rho_max - rho_min) / rho_step)))
    psi_step = 2.0 * math.pi / int(psi_bins)
    psi_shifted = (psi + math.pi) % (2.0 * math.pi)

    out = pd.DataFrame(
        {
            "x_m": x,
            "rho_m": rho,
            "psi_rad": psi,
            "x_bin": _bin_index(x, x_min, x_step, n_x),
            "rho_bin": _bin_index(rho, rho_min, rho_step, n_rho),
            "psi_bin": _bin_index(psi_shifted, 0.0, psi_step, int(psi_bins)),
        }
    )
    meta = {
        "x_min": x_min,
        "x_max": x_max,
        "rho_min": rho_min,
        "rho_max": rho_max,
        "x_step_m": x_step,
        "rho_step_m": rho_step,
        "psi_bins": int(psi_bins),
        "n_x_bins": int(n_x),
        "n_rho_bins": int(n_rho),
    }
    return out, meta


def diagnose(df: pd.DataFrame, *, x_bin_mm: float, rho_bin_mm: float, psi_bins: int) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
    sf, meta = shell_frame(df, x_bin_mm=x_bin_mm, rho_bin_mm=rho_bin_mm, psi_bins=psi_bins)
    xpsi = sf.groupby(["x_bin", "psi_bin"], sort=False).agg(
        count=("rho_m", "size"),
        rho_min=("rho_m", "min"),
        rho_max=("rho_m", "max"),
    )
    xpsi = xpsi.reset_index()
    xpsi["radial_thickness_m"] = xpsi["rho_max"] - xpsi["rho_min"]

    angular = sf.groupby("x_bin")["psi_bin"].nunique().reset_index(name="covered_psi_bins")
    angular["angular_completeness"] = angular["covered_psi_bins"] / float(psi_bins)

    occupied_3d = sf.drop_duplicates(["x_bin", "psi_bin", "rho_bin"]).shape[0]
    possible_3d = int(meta["n_x_bins"]) * int(psi_bins) * int(meta["n_rho_bins"])
    shell_hole_ratio = 1.0 - float(occupied_3d) / max(1.0, float(possible_3d))

    pivot_min = xpsi.pivot(index="x_bin", columns="psi_bin", values="rho_min")
    pivot_max = xpsi.pivot(index="x_bin", columns="psi_bin", values="rho_max")
    d1_min = np.diff(pivot_min.to_numpy(dtype=float), axis=0)
    d1_max = np.diff(pivot_max.to_numpy(dtype=float), axis=0)
    d2_min = np.diff(d1_min, axis=0)
    d2_max = np.diff(d1_max, axis=0)

    def p95_abs(arr: np.ndarray) -> float:
        vals = np.abs(arr[np.isfinite(arr)])
        return float(np.percentile(vals, 95)) if len(vals) else float("nan")

    report = {
        "rows": int(len(df)),
        "x_range_m": [float(sf["x_m"].min()), float(sf["x_m"].max())],
        "rho_range_m": [float(sf["rho_m"].min()), float(sf["rho_m"].max())],
        "angular_completeness_mean": float(angular["angular_completeness"].mean()),
        "angular_completeness_p10": float(np.percentile(angular["angular_completeness"], 10)),
        "angular_completeness_min": float(angular["angular_completeness"].min()),
        "radial_thickness_m_mean": float(xpsi["radial_thickness_m"].mean()),
        "radial_thickness_m_p10": float(np.percentile(xpsi["radial_thickness_m"], 10)),
        "radial_thickness_m_p50": float(np.percentile(xpsi["radial_thickness_m"], 50)),
        "radial_thickness_m_p90": float(np.percentile(xpsi["radial_thickness_m"], 90)),
        "shell_hole_ratio_bbox": float(shell_hole_ratio),
        "occupied_xpsi_bins": int(len(xpsi)),
        "occupied_3d_bins": int(occupied_3d),
        "possible_3d_bins_bbox": int(possible_3d),
        "envelope_smoothness": {
            "rho_min_d1_p95_m": p95_abs(d1_min),
            "rho_max_d1_p95_m": p95_abs(d1_max),
            "rho_min_d2_p95_m": p95_abs(d2_min),
            "rho_max_d2_p95_m": p95_abs(d2_max),
        },
        "binning": meta,
    }
    frames = {"shell_points": sf, "xpsi": xpsi, "angular": angular}
    return report, frames, meta


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_report(dataset_id: str, report: dict[str, Any], frames: dict[str, pd.DataFrame], meta: dict[str, Any], out_dir: Path) -> None:
    heat_dir = out_dir / f"{dataset_id}_shell_heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)
    sf = frames["shell_points"]
    xpsi = frames["xpsi"]
    angular = frames["angular"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sf["x_m"], sf["rho_m"], s=2, alpha=0.25, color="#235789")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("rho=sqrt(y^2+z^2) (m)")
    ax.set_title(f"{dataset_id}: x-rho envelope")
    _savefig(fig, heat_dir / "x_rho_envelope.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(meta["x_min"] + angular["x_bin"] * meta["x_step_m"], angular["angular_completeness"], lw=1.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("angular completeness")
    ax.set_title(f"{dataset_id}: x-psi angular completeness")
    _savefig(fig, heat_dir / "x_psi_angular_completeness.png")

    count_grid = np.zeros((int(meta["n_x_bins"]), int(meta["psi_bins"])), dtype=float)
    thick_grid = np.full_like(count_grid, np.nan)
    for row in xpsi.itertuples(index=False):
        count_grid[int(row.x_bin), int(row.psi_bin)] = float(row.count)
        thick_grid[int(row.x_bin), int(row.psi_bin)] = float(row.radial_thickness_m) * 1000.0

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(np.log10(count_grid.T + 1.0), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("x bin")
    ax.set_ylabel("psi bin")
    ax.set_title(f"{dataset_id}: log10 occupancy")
    fig.colorbar(im, ax=ax, label="log10(count+1)")
    _savefig(fig, heat_dir / "x_psi_occupancy.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(thick_grid.T, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("x bin")
    ax.set_ylabel("psi bin")
    ax.set_title(f"{dataset_id}: radial thickness (mm)")
    fig.colorbar(im, ax=ax, label="rho max-min (mm)")
    _savefig(fig, heat_dir / "x_psi_radial_thickness.png")

    (out_dir / f"{dataset_id}_shell_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def write_summary(out_dir: Path, reports: dict[str, dict[str, Any]]) -> None:
    rows = []
    for dataset_id, report in reports.items():
        rows.append(
            {
                "dataset_id": dataset_id,
                "rows": report["rows"],
                "x_min": report["x_range_m"][0],
                "x_max": report["x_range_m"][1],
                "rho_min": report["rho_range_m"][0],
                "rho_max": report["rho_range_m"][1],
                "angular_mean": report["angular_completeness_mean"],
                "angular_p10": report["angular_completeness_p10"],
                "radial_thickness_p10_mm": report["radial_thickness_m_p10"] * 1000.0,
                "radial_thickness_p50_mm": report["radial_thickness_m_p50"] * 1000.0,
                "radial_thickness_p90_mm": report["radial_thickness_m_p90"] * 1000.0,
                "shell_hole_ratio_bbox": report["shell_hole_ratio_bbox"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "shell_summary.csv", index=False)
    lines = [
        "# Shell completeness diagnostics",
        "",
        "| dataset | rows | x range | rho range | angular mean | angular p10 | radial p10 mm | radial p50 mm | shell hole ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset_id} | {rows} | {xmin:.3f}-{xmax:.3f} | {rmin:.3f}-{rmax:.3f} | {amean:.3f} | {ap10:.3f} | {tp10:.2f} | {tp50:.2f} | {hole:.3f} |".format(
                dataset_id=row["dataset_id"],
                rows=row["rows"],
                xmin=row["x_min"],
                xmax=row["x_max"],
                rmin=row["rho_min"],
                rmax=row["rho_max"],
                amean=row["angular_mean"],
                ap10=row["angular_p10"],
                tp10=row["radial_thickness_p10_mm"],
                tp50=row["radial_thickness_p50_mm"],
                hole=row["shell_hole_ratio_bbox"],
            )
        )
    (out_dir / "shell_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for dataset_id in [v.strip() for v in str(args.datasets).split(",") if v.strip()]:
        if dataset_id not in DEFAULT_DATASETS:
            raise ValueError(f"unknown dataset {dataset_id}; available={sorted(DEFAULT_DATASETS)}")
        df = pd.read_parquet(DEFAULT_DATASETS[dataset_id], columns=XYZ_COLS)
        if int(args.max_rows) > 0 and len(df) > int(args.max_rows):
            df = df.sample(n=int(args.max_rows), random_state=int(args.seed)).reset_index(drop=True)
        report, frames, meta = diagnose(df, x_bin_mm=float(args.x_bin_mm), rho_bin_mm=float(args.rho_bin_mm), psi_bins=int(args.psi_bins))
        reports[dataset_id] = report
        plot_report(dataset_id, report, frames, meta, out_dir)
    write_summary(out_dir, reports)
    payload = {"mode": "shell_completeness_diagnostics", "out_dir": str(out_dir), "reports": reports}
    (out_dir / "shell_reports.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diagnose canonical shell completeness in x-psi-rho coordinates.")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "02_shell_completeness_diagnostics")
    ap.add_argument("--datasets", default="path_a,path_d")
    ap.add_argument("--x-bin-mm", type=float, default=5.0)
    ap.add_argument("--rho-bin-mm", type=float, default=5.0)
    ap.add_argument("--psi-bins", type=int, default=72)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260707)
    return ap.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({"out_dir": payload["out_dir"], "datasets": list(payload["reports"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
