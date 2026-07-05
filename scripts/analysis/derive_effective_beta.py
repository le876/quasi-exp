#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quasi_exp.model.sampling import beta_to_theta, effective_beta_from_theta  # noqa: E402


EFFECTIVE_BETA_COLS = [f"effective_beta_{i}_rad" for i in range(1, 7)]


def _numbered_cols(columns: list[str], prefix: str, suffix: str) -> list[str]:
    out: list[tuple[int, str]] = []
    for col in columns:
        if not (col.startswith(prefix) and col.endswith(suffix)):
            continue
        raw = col[len(prefix) : -len(suffix)]
        if raw.isdigit():
            out.append((int(raw), col))
    return [col for _, col in sorted(out)]


def append_effective_beta(dataset: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    theta_cols = _numbered_cols(list(dataset.columns), "theta_", "_rad")
    if len(theta_cols) != 30:
        raise ValueError(f"expected 30 theta columns, got {len(theta_cols)}")
    theta = dataset[theta_cols].to_numpy(dtype=float)
    beta = effective_beta_from_theta(theta)
    out = meta.copy()
    for i, col in enumerate(EFFECTIVE_BETA_COLS):
        out[col] = beta[:, i]
    out["effective_beta_source"] = "theta_odd_even_mean"
    theta_roundtrip = np.vstack([beta_to_theta(row) for row in beta])
    report = {
        "rows": int(len(out)),
        "effective_beta_source": "theta_odd_even_mean",
        "theta_roundtrip_mae_rad": float(np.mean(np.abs(theta_roundtrip - theta))),
        "theta_roundtrip_max_abs_rad": float(np.max(np.abs(theta_roundtrip - theta))),
    }
    return out, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-meta", required=True, type=Path)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    dataset = pd.read_parquet(args.dataset)
    meta = pd.read_parquet(args.meta)
    out_meta, report = append_effective_beta(dataset, meta)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.to_parquet(args.out_meta, index=False)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
