#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _quantiles(x: np.ndarray, qs: list[float]) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    out: dict[str, float] = {}
    for q in qs:
        out[f"q{int(q*100):02d}"] = float(np.quantile(x, q))
    out["min"] = float(np.min(x))
    out["max"] = float(np.max(x))
    out["mean"] = float(np.mean(x))
    return out


def _find_cols(cols: list[str], prefix: str) -> list[str]:
    return [c for c in cols if c.startswith(prefix)]


def analyze(dataset_path: Path, meta_path: Path | None, tension_cap_n: float, x_bins: int, r_bins: int):
    table = pq.read_table(dataset_path)
    df = table.to_pandas()

    cols = list(df.columns)
    theta_cols = [c for c in cols if c.startswith("theta_") and c.endswith("_rad")]
    tension_cols = [c for c in cols if c.startswith("tension_") and c.endswith("_n")]

    if not {"x_m", "y_m", "z_m"}.issubset(cols):
        raise SystemExit(f"dataset missing x_m/y_m/z_m, got cols={cols[:10]}...")
    if not theta_cols:
        raise SystemExit("dataset missing theta_*_rad columns")
    if not tension_cols:
        raise SystemExit("dataset missing tension_*_n columns")

    x = df["x_m"].to_numpy(dtype=float)
    y = df["y_m"].to_numpy(dtype=float)
    z = df["z_m"].to_numpy(dtype=float)
    r_yz = np.sqrt(y**2 + z**2)

    theta = df[theta_cols].to_numpy(dtype=float)
    max_abs_theta = np.max(np.abs(theta), axis=1)

    T = df[tension_cols].to_numpy(dtype=float)
    max_t = np.max(T, axis=1)
    sat_mask = T >= (tension_cap_n - 1e-9)
    sat_count = np.sum(sat_mask, axis=1)

    report: dict[str, object] = {}
    report["rows"] = int(len(df))
    report["cols"] = int(len(cols))
    report["dataset_path"] = str(dataset_path)
    report["tension_cap_n"] = float(tension_cap_n)

    report["workspace_quantiles"] = {
        "x_m": _quantiles(x, [0.5, 0.9, 0.95, 0.99]),
        "y_m": _quantiles(y, [0.5, 0.9, 0.95, 0.99]),
        "z_m": _quantiles(z, [0.5, 0.9, 0.95, 0.99]),
        "r_yz_m": _quantiles(r_yz, [0.5, 0.9, 0.95, 0.99]),
    }

    report["theta_max_abs"] = {
        "rad": _quantiles(max_abs_theta, [0.5, 0.9, 0.95, 0.99]),
        "deg": _quantiles(max_abs_theta * 180.0 / np.pi, [0.5, 0.9, 0.95, 0.99]),
    }

    report["tension_max"] = _quantiles(max_t, [0.5, 0.9, 0.95, 0.99])
    report["sat_ratio_max_t"] = float(np.mean(max_t >= (tension_cap_n - 1e-9)))

    # histogram of number of saturated cables per sample
    vals, cnts = np.unique(sat_count, return_counts=True)
    report["sat_count_hist"] = {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}

    # per-cable saturation ratios
    per_cable = {c: float(np.mean(df[c].to_numpy(dtype=float) >= (tension_cap_n - 1e-9))) for c in tension_cols}
    report["per_cable_sat_ratio"] = dict(sorted(per_cable.items(), key=lambda kv: kv[1], reverse=True))

    # workspace occupancy (x bins × r bins)
    x_edges = np.linspace(np.min(x), np.max(x), x_bins + 1)
    r_edges = np.linspace(np.min(r_yz), np.max(r_yz), r_bins + 1)
    H, _, _ = np.histogram2d(x, r_yz, bins=[x_edges, r_edges])
    report["workspace_hist2d"] = {
        "x_edges_m": x_edges.tolist(),
        "r_edges_m": r_edges.tolist(),
        "counts": H.astype(int).tolist(),
    }

    if meta_path is not None and meta_path.exists():
        meta = pq.read_table(meta_path).to_pandas()
        if "rms_rnorm" in meta.columns:
            report["rms_rnorm"] = _quantiles(meta["rms_rnorm"].to_numpy(dtype=float), [0.5, 0.9, 0.95, 0.99])
        if "max_tension" in meta.columns:
            report["meta_max_tension"] = _quantiles(meta["max_tension"].to_numpy(dtype=float), [0.5, 0.9, 0.95, 0.99])
        report["meta_path"] = str(meta_path)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path, help="dataset.parquet")
    ap.add_argument("--meta", type=Path, default=None, help="dataset_meta.parquet (optional)")
    ap.add_argument("--tension-cap", type=float, default=2000.0)
    ap.add_argument("--x-bins", type=int, default=12)
    ap.add_argument("--r-bins", type=int, default=12)
    ap.add_argument("--out-json", type=Path, default=None, help="Write report JSON to this path (optional)")
    args = ap.parse_args()

    report = analyze(
        dataset_path=args.dataset,
        meta_path=args.meta,
        tension_cap_n=args.tension_cap,
        x_bins=args.x_bins,
        r_bins=args.r_bins,
    )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

