#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_priority_grid_dataset import (  # noqa: E402
    BETA_COLS,
    TARGET_COLS,
    _json_default,
    _label_beta,
    _load_script,
    attach_workspace_targets,
    diagnose_and_gate,
    workspace_coverage,
)
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from quasi_exp.model.quasi_static import QuasiStaticModel  # noqa: E402


POLICY_VERSION = "priority_grid_fixed_layer_manifold_v1"
LAYER_PAIRS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 0.25),
    (0.0, 0.5),
    (0.125, 0.25),
    (0.125, 0.5),
    (0.25, 0.25),
    (0.25, 0.5),
)


def layer_label(s1: float, s2: float) -> str:
    return f"s1_{int(round(float(s1) * 1000)):04d}_s2_{int(round(float(s2) * 1000)):04d}"


def _unit_sobol(n: int, d: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, d), dtype=float)
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=d, scramble=True, seed=int(seed))
        m = int(np.ceil(np.log2(max(1, n))))
        return np.asarray(sampler.random_base2(m), dtype=float)[:n]
    except Exception:
        return np.random.default_rng(int(seed)).random((n, d))


def _unit_lhs(n: int, d: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, d), dtype=float)
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=d, seed=int(seed))
        return np.asarray(sampler.random(n), dtype=float)
    except Exception:
        rng = np.random.default_rng(int(seed))
        edges = np.linspace(0.0, 1.0, n + 1)
        out = np.zeros((n, d), dtype=float)
        for col in range(d):
            out[:, col] = edges[:-1] + rng.random(n) / n
            rng.shuffle(out[:, col])
        return out


def _scale_unit(unit: np.ndarray, bounds_rad: tuple[tuple[float, float], tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(unit, dtype=float).reshape(-1, 2)
    lo = np.asarray([bounds_rad[0][0], bounds_rad[1][0]], dtype=float)
    hi = np.asarray([bounds_rad[0][1], bounds_rad[1][1]], dtype=float)
    return lo.reshape(1, 2) + np.clip(arr, 0.0, 1.0) * (hi - lo).reshape(1, 2)


def _source_counts(num_samples: int) -> tuple[int, int, int]:
    n = int(num_samples)
    if n <= 0:
        raise ValueError("num_samples must be > 0")
    grid = min(n, int(round(n * (1681.0 / 2000.0))))
    remaining = n - grid
    sobol = int(np.ceil(remaining / 2.0))
    lhs = remaining - sobol
    return grid, sobol, lhs


def _grid_points(n: int, beta5_rad: tuple[float, float], beta6_rad: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    levels = int(np.ceil(np.sqrt(n)))
    b5 = np.linspace(float(beta5_rad[0]), float(beta5_rad[1]), levels, dtype=float)
    b6 = np.linspace(float(beta6_rad[0]), float(beta6_rad[1]), levels, dtype=float)
    rows: list[tuple[float, float]] = []
    i5s: list[int] = []
    i6s: list[int] = []
    for i5, v5 in enumerate(b5):
        for i6, v6 in enumerate(b6):
            if len(rows) >= n:
                break
            rows.append((float(v5), float(v6)))
            i5s.append(i5)
            i6s.append(i6)
        if len(rows) >= n:
            break
    return np.asarray(rows, dtype=float), np.asarray(i5s, dtype=int), np.asarray(i6s, dtype=int)


def _rows_from_beta56(
    beta56: np.ndarray,
    *,
    s1: float,
    s2: float,
    sample_source: str,
    start_id: int,
    grid_i5: np.ndarray | None = None,
    grid_i6: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    label = layer_label(s1, s2)
    out: list[dict[str, Any]] = []
    for local_idx, pair in enumerate(np.asarray(beta56, dtype=float).reshape(-1, 2)):
        beta5, beta6 = float(pair[0]), float(pair[1])
        beta = np.asarray([s1 * beta5, s1 * beta6, s2 * beta5, s2 * beta6, beta5, beta6], dtype=float)
        row: dict[str, Any] = {
            "priority_grid_id": int(start_id + local_idx),
            "priority_ratio_pair_id": 0,
            "s1": float(s1),
            "s2": float(s2),
            "layer_label": label,
            "sample_source": str(sample_source),
            "source_component": label,
            "priority_policy_version": POLICY_VERSION,
            "beta5_grid_index": int(grid_i5[local_idx]) if grid_i5 is not None else -1,
            "beta6_grid_index": int(grid_i6[local_idx]) if grid_i6 is not None else -1,
            "beta5_grid_deg": float(np.rad2deg(beta5)),
            "beta6_grid_deg": float(np.rad2deg(beta6)),
        }
        for j, value in enumerate(beta):
            row[f"beta{j + 1}_rad"] = float(value)
        out.append(row)
    return out


def build_fixed_layer_pool(
    *,
    s1: float,
    s2: float,
    num_samples: int = 2000,
    beta5_deg: tuple[float, float] = (-10.0, 10.0),
    beta6_deg: tuple[float, float] = (-10.0, 10.0),
    seed: int = 20260614,
) -> pd.DataFrame:
    if float(s1) > float(s2) + 1.0e-12:
        raise ValueError("fixed-layer policy requires s1 <= s2")
    beta5_rad = tuple(float(v) for v in np.deg2rad(beta5_deg))
    beta6_rad = tuple(float(v) for v in np.deg2rad(beta6_deg))
    grid_n, sobol_n, lhs_n = _source_counts(int(num_samples))

    rows: list[dict[str, Any]] = []
    grid, i5, i6 = _grid_points(grid_n, beta5_rad, beta6_rad)
    rows.extend(_rows_from_beta56(grid, s1=s1, s2=s2, sample_source="grid", start_id=len(rows), grid_i5=i5, grid_i6=i6))

    sobol = _scale_unit(_unit_sobol(sobol_n, 2, int(seed) + 17), (beta5_rad, beta6_rad))
    rows.extend(_rows_from_beta56(sobol, s1=s1, s2=s2, sample_source="sobol", start_id=len(rows)))

    lhs = _scale_unit(_unit_lhs(lhs_n, 2, int(seed) + 31), (beta5_rad, beta6_rad))
    rows.extend(_rows_from_beta56(lhs, s1=s1, s2=s2, sample_source="lhs", start_id=len(rows)))

    return pd.DataFrame(rows).reset_index(drop=True)


def generate_fixed_layer_dataset(
    *,
    cfg: dict[str, Any],
    out_dir: Path,
    s1: float,
    s2: float,
    num_samples: int = 2000,
    beta5_deg: tuple[float, float] = (-10.0, 10.0),
    beta6_deg: tuple[float, float] = (-10.0, 10.0),
    seed: int = 20260614,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_robot_inputs(cfg)
    model = QuasiStaticModel(cfg, inputs)
    t0 = time.time()

    pool = build_fixed_layer_pool(
        s1=float(s1),
        s2=float(s2),
        num_samples=int(num_samples),
        beta5_deg=tuple(beta5_deg),
        beta6_deg=tuple(beta6_deg),
        seed=int(seed),
    )
    pool = attach_workspace_targets(pool, inputs, model.theta_sign)
    pool.to_parquet(out_dir / "fixed_layer_pool.parquet", index=False)

    pso_cfg = cfg["pso"]
    rms_thresh = float(cfg.get("dataset", {}).get("rms_rnorm_threshold", 0.06))
    rows: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    failures = 0
    for sample_id, pool_row in enumerate(tqdm(pool.itertuples(index=False), total=len(pool), desc=layer_label(s1, s2))):
        series = pd.Series(pool_row._asdict())
        beta = np.asarray([getattr(pool_row, f"beta{i}_rad") for i in range(1, 7)], dtype=float)
        ok, row, meta = _label_beta(
            sample_id=sample_id,
            beta=beta,
            pool_row=series,
            model=model,
            inputs=inputs,
            pso_cfg=pso_cfg,
            pso_seed=int(pso_cfg.get("rng_seed", seed)) + sample_id,
            rms_thresh=rms_thresh,
        )
        if not ok:
            failures += 1
            continue
        meta["source_component"] = str(pool_row.layer_label)
        meta["layer_label"] = str(pool_row.layer_label)
        meta["sample_source"] = str(pool_row.sample_source)
        meta["friction_case_signature"] = "".join(str(int(meta.get(f"case_{j}", 0))) for j in range(1, 13))
        rows.append(row)
        metas.append(meta)

    dataset = pd.DataFrame(rows).reset_index(drop=True)
    meta = pd.DataFrame(metas).reset_index(drop=True)
    if len(dataset):
        dataset["sample_id"] = np.arange(len(dataset), dtype=int)
        meta["sample_id"] = np.arange(len(meta), dtype=int)
    dataset.to_parquet(out_dir / "dataset.parquet", index=False)
    meta.to_parquet(out_dir / "dataset_meta.parquet", index=False)

    diagnostics = diagnose_and_gate(dataset, meta, out_dir / "diagnostics") if len(dataset) else {}
    report = {
        "mode": "fixed_layer_manifold",
        "policy_version": POLICY_VERSION,
        "layer_label": layer_label(s1, s2),
        "s1": float(s1),
        "s2": float(s2),
        "requested_rows": int(len(pool)),
        "selected_rows": int(len(dataset)),
        "label_failures": int(failures),
        "elapsed_s": float(time.time() - t0),
        "sec_per_selected_row": float((time.time() - t0) / max(len(dataset), 1)),
        "source_counts": {str(k): int(v) for k, v in pool["sample_source"].value_counts().sort_index().to_dict().items()},
        "workspace_coverage": workspace_coverage(pool[TARGET_COLS].to_numpy(dtype=float)),
        "gate": diagnostics.get("gate", {}),
    }
    (out_dir / "fixed_layer_generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return report


def _parse_pair(raw: str) -> tuple[float, float]:
    parts = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated floats")
    return float(parts[0]), float(parts[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--s1", required=True, type=float)
    ap.add_argument("--s2", required=True, type=float)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--beta5-deg", type=_parse_pair, default=(-10.0, 10.0))
    ap.add_argument("--beta6-deg", type=_parse_pair, default=(-10.0, 10.0))
    ap.add_argument("--seed", type=int, default=20260614)
    args = ap.parse_args()
    cfg = load_config(args.config)
    report = generate_fixed_layer_dataset(
        cfg=cfg,
        out_dir=args.out_dir,
        s1=float(args.s1),
        s2=float(args.s2),
        num_samples=int(args.num_samples),
        beta5_deg=tuple(args.beta5_deg),
        beta6_deg=tuple(args.beta6_deg),
        seed=int(args.seed),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
