#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-metrics", required=True, type=Path, help="path to all_metrics.json")
    args = ap.parse_args()

    data = json.loads(args.all_metrics.read_text(encoding="utf-8"))
    keys = sorted(data.keys())

    cols = [
        ("theta_mae_deg", "θ MAE (deg)"),
        ("theta_rmse_deg", "θ RMSE (deg)"),
        ("tension_mae_n", "T MAE (N)"),
        ("tension_rmse_n", "T RMSE (N)"),
        ("tension_lt0_ratio", "%(T<0)"),
        ("tension_gt_tmax_ratio", "%(T>tmax)"),
        ("ee_pos_p95_mm", "EE p95 (mm)"),
        ("pred_time_ms_per_sample", "Pred ms/sample"),
        ("train_rows_used", "Train rows"),
    ]

    # markdown
    header = ["model"] + [c[1] for c in cols]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, (int,)):
            return str(v)
        try:
            fv = float(v)
        except Exception:
            return str(v)
        # ratios
        if "ratio" in str(v):
            return "%.3f" % fv
        if abs(fv) >= 1000:
            return "%.1f" % fv
        return "%.4g" % fv

    for k in keys:
        row = [k]
        for c, _title in cols:
            v = data[k].get(c, None)
            if c.endswith("_ratio") and v is not None:
                v = 100.0 * float(v)
            row.append(fmt(v))
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()

