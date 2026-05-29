#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _read_parquet(path):
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pandas()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path, help="dataset.parquet")
    ap.add_argument("--out", required=True, type=Path, help="output .npz path")
    args = ap.parse_args()

    df = _read_parquet(args.dataset)
    cols = list(df.columns)
    theta_cols = sorted([c for c in cols if c.startswith("theta_") and c.endswith("_rad")], key=lambda c: int(c.split("_")[1]))
    tension_cols = sorted([c for c in cols if c.startswith("tension_") and c.endswith("_n")], key=lambda c: int(c.split("_")[1]))
    if len(theta_cols) != 30 or len(tension_cols) != 12:
        raise SystemExit("unexpected columns; expected 30 theta_*_rad and 12 tension_*_n")

    X = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float32)
    theta = df[theta_cols].to_numpy(dtype=np.float32)
    tension = df[tension_cols].to_numpy(dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x_m=X,
        theta_rad=theta,
        tension_n=tension,
        theta_cols=np.array(theta_cols, dtype=object),
        tension_cols=np.array(tension_cols, dtype=object),
    )
    print("OK:", args.out, "rows=", X.shape[0])


if __name__ == "__main__":
    main()

