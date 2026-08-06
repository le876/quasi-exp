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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "runs" / ".mplconfig"))

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
XYZ_COLS = ["x_m", "y_m", "z_m"]
U_COLS = ["u_a_deg", "u_b_deg", "u_eta", "u_a_rad", "u_b_rad", "rho_deg", "s1", "s2"]
JAC_COLS = ["sigma1_m", "sigma2_m", "sigma3_m", "kappa", "jacobian_gate_pass"]


DATASETS = {
    "path_a_full": {
        "path": "data/canonical_layer_field_u3_pilot_v1/path_a_sync_plus/full_pool.parquet",
        "source_type": "canonical_path_a_full",
        "expected_path_name": "path_a_sync_plus",
    },
    "path_a_x": {
        "path": "data/canonical_layer_field_u3_pilot_v1/path_a_sync_plus/x1p0_1p2_pool.parquet",
        "source_type": "canonical_path_a_x",
        "expected_path_name": "path_a_sync_plus",
    },
    "path_a_jac": {
        "path": "data/canonical_layer_field_u3_pilot_v1/path_a_sync_plus/jacobian_pass_pool.parquet",
        "source_type": "canonical_path_a_jacobian",
        "expected_path_name": "path_a_sync_plus",
    },
    "path_a_balanced": {
        "path": "data/canonical_layer_field_u3_pilot_v1/path_a_sync_plus/balanced_20000.parquet",
        "source_type": "canonical_path_a_balanced",
        "expected_path_name": "path_a_sync_plus",
    },
    "path_d_full": {
        "path": "data/canonical_layer_field_u3_pilot_v1/path_d_wide_redistribute/full_pool.parquet",
        "source_type": "canonical_path_d_full",
        "expected_path_name": "path_d_wide_redistribute",
    },
    "path_d_x": {
        "path": "data/canonical_layer_field_u3_pilot_v1/path_d_wide_redistribute/x1p0_1p2_pool.parquet",
        "source_type": "canonical_path_d_x",
        "expected_path_name": "path_d_wide_redistribute",
    },
    "path_d_jac": {
        "path": "data/canonical_layer_field_u3_pilot_v1/path_d_wide_redistribute/jacobian_pass_pool.parquet",
        "source_type": "canonical_path_d_jacobian",
        "expected_path_name": "path_d_wide_redistribute",
    },
    "path_d_balanced": {
        "path": "data/canonical_layer_field_u3_pilot_v1/path_d_wide_redistribute/balanced_20000.parquet",
        "source_type": "canonical_path_d_balanced",
        "expected_path_name": "path_d_wide_redistribute",
    },
    "hierarchical_full": {
        "path": "data/hierarchical_beta_fk_x1p0_1p2_v1/full_fk_pool.parquet",
        "source_type": "hierarchical_full",
        "expected_path_name": "",
    },
    "hierarchical_x": {
        "path": "data/hierarchical_beta_fk_x1p0_1p2_v1/x1p0_1p2_pool.parquet",
        "source_type": "hierarchical_x",
        "expected_path_name": "",
    },
    "hierarchical_balanced": {
        "path": "data/hierarchical_beta_fk_x1p0_1p2_v1/balanced_500k.parquet",
        "source_type": "hierarchical_balanced",
        "expected_path_name": "",
    },
}


@dataclass
class DatasetAudit:
    dataset_id: str
    path: str
    source_type: str
    exists: bool
    rows: int
    columns: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    has_u: bool
    has_beta6: bool
    has_theta30: bool
    has_jacobian: bool
    has_source_component: bool
    path_name_consistent: bool | None
    fk_recheck_p95_m: float | None
    fk_recheck_max_m: float | None
    schema_missing_required: str


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value.as_py() if hasattr(value, "as_py") else value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _min_max(pf: pq.ParquetFile, col: str) -> tuple[float, float]:
    mins: list[float] = []
    maxs: list[float] = []
    for batch in pf.iter_batches(columns=[col], batch_size=256_000):
        arr = batch.column(0)
        if len(arr) == 0:
            continue
        mm = pc.min_max(arr)
        mins.append(_finite_or_nan(mm["min"]))
        maxs.append(_finite_or_nan(mm["max"]))
    if not mins:
        return float("nan"), float("nan")
    return float(np.nanmin(mins)), float(np.nanmax(maxs))


def _sample_rows(path: Path, columns: list[str], *, sample_rows: int, seed: int) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    rows = int(pf.metadata.num_rows)
    if rows <= 0:
        return pd.DataFrame(columns=columns)
    rng = np.random.default_rng(int(seed))
    row_groups = np.arange(pf.num_row_groups, dtype=np.int64)
    rng.shuffle(row_groups)
    chunks: list[pd.DataFrame] = []
    got = 0
    for rg in row_groups:
        table = pf.read_row_group(int(rg), columns=columns)
        df = table.to_pandas()
        if len(df) == 0:
            continue
        take = min(len(df), int(sample_rows) - got)
        if take < len(df):
            idx = rng.choice(len(df), size=take, replace=False)
            df = df.iloc[np.sort(idx)]
        chunks.append(df)
        got += len(df)
        if got >= int(sample_rows):
            break
    if not chunks:
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True).iloc[: int(sample_rows)]


def _path_name_consistency(path: Path, cols: list[str], expected: str, *, sample_rows: int, seed: int) -> bool | None:
    if "path_name" not in cols:
        return None
    if not expected:
        return None
    df = _sample_rows(path, ["path_name"], sample_rows=min(int(sample_rows), 5000), seed=seed)
    vals = set(str(v) for v in df["path_name"].dropna().unique().tolist())
    return vals == {str(expected)}


def _fk_recheck(
    path: Path,
    cols: list[str],
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    sample_rows: int,
    seed: int,
) -> tuple[float | None, float | None]:
    needed = THETA_COLS + XYZ_COLS
    if any(c not in cols for c in needed):
        return None, None
    df = _sample_rows(path, needed, sample_rows=sample_rows, seed=seed)
    if df.empty:
        return None, None
    theta = df[THETA_COLS].to_numpy(dtype=float)
    xyz = df[XYZ_COLS].to_numpy(dtype=float)
    pred = fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    err = np.linalg.norm(pred - xyz, axis=1)
    return float(np.percentile(err, 95)), float(np.max(err))


def audit_dataset(
    dataset_id: str,
    spec: dict[str, str],
    *,
    repo_root: Path,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    sample_rows: int,
    seed: int,
) -> DatasetAudit:
    rel_path = spec["path"]
    path = repo_root / rel_path
    if not path.exists():
        return DatasetAudit(
            dataset_id=dataset_id,
            path=rel_path,
            source_type=spec["source_type"],
            exists=False,
            rows=0,
            columns=0,
            x_min=float("nan"),
            x_max=float("nan"),
            y_min=float("nan"),
            y_max=float("nan"),
            z_min=float("nan"),
            z_max=float("nan"),
            has_u=False,
            has_beta6=False,
            has_theta30=False,
            has_jacobian=False,
            has_source_component=False,
            path_name_consistent=None,
            fk_recheck_p95_m=None,
            fk_recheck_max_m=None,
            schema_missing_required="file_missing",
        )
    pf = pq.ParquetFile(path)
    cols = pf.schema_arrow.names
    colset = set(cols)
    ranges = {}
    for col in XYZ_COLS:
        if col in colset:
            ranges[col] = _min_max(pf, col)
        else:
            ranges[col] = (float("nan"), float("nan"))

    is_canonical = spec["source_type"].startswith("canonical")
    required = XYZ_COLS + BETA_COLS + THETA_COLS
    if is_canonical:
        required += U_COLS
    if "jacobian" in spec["source_type"]:
        required += JAC_COLS
    missing = [c for c in required if c not in colset]

    p95, max_err = _fk_recheck(
        path,
        cols,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        sample_rows=sample_rows,
        seed=seed,
    )
    return DatasetAudit(
        dataset_id=dataset_id,
        path=rel_path,
        source_type=spec["source_type"],
        exists=True,
        rows=int(pf.metadata.num_rows),
        columns=len(cols),
        x_min=ranges["x_m"][0],
        x_max=ranges["x_m"][1],
        y_min=ranges["y_m"][0],
        y_max=ranges["y_m"][1],
        z_min=ranges["z_m"][0],
        z_max=ranges["z_m"][1],
        has_u=all(c in colset for c in U_COLS),
        has_beta6=all(c in colset for c in BETA_COLS),
        has_theta30=all(c in colset for c in THETA_COLS),
        has_jacobian=all(c in colset for c in JAC_COLS),
        has_source_component="source_component" in colset,
        path_name_consistent=_path_name_consistency(
            path,
            cols,
            str(spec.get("expected_path_name", "")),
            sample_rows=sample_rows,
            seed=seed,
        ),
        fk_recheck_p95_m=p95,
        fk_recheck_max_m=max_err,
        schema_missing_required=",".join(missing),
    )


def write_schema_report(out_path: Path, rows: list[DatasetAudit], *, fk_tol_m: float) -> None:
    lines = [
        "# Large ellipse data audit",
        "",
        f"- FK tolerance p95: `{fk_tol_m:g} m`",
        f"- Datasets checked: `{len(rows)}`",
        "",
        "| dataset | rows | has u | has beta6 | has theta30 | has jac | path ok | fk p95 m | missing required |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {dataset} | {rows} | {has_u} | {has_beta6} | {has_theta30} | {has_jacobian} | {path_ok} | {fk_p95} | {missing} |".format(
                dataset=r.dataset_id,
                rows=r.rows,
                has_u="yes" if r.has_u else "no",
                has_beta6="yes" if r.has_beta6 else "no",
                has_theta30="yes" if r.has_theta30 else "no",
                has_jacobian="yes" if r.has_jacobian else "no",
                path_ok="n/a" if r.path_name_consistent is None else ("yes" if r.path_name_consistent else "no"),
                fk_p95="n/a" if r.fk_recheck_p95_m is None else f"{r.fk_recheck_p95_m:.3e}",
                missing=r.schema_missing_required or "",
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(str(args.robot_config))
    inputs = load_robot_inputs(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audits = [
        audit_dataset(
            dataset_id,
            spec,
            repo_root=REPO_ROOT,
            lengths_m=inputs.lengths_m,
            p_end_local_m=inputs.p_end_local_m,
            sample_rows=int(args.sample_rows),
            seed=int(args.seed) + i * 17,
        )
        for i, (dataset_id, spec) in enumerate(DATASETS.items())
    ]
    df = pd.DataFrame([asdict(r) for r in audits])
    df.to_csv(out_dir / "data_manifest.csv", index=False)
    write_schema_report(out_dir / "schema_report.md", audits, fk_tol_m=float(args.fk_tol_m))

    fk_rows = [
        {
            "dataset_id": r.dataset_id,
            "fk_recheck_p95_m": r.fk_recheck_p95_m,
            "fk_recheck_max_m": r.fk_recheck_max_m,
            "fk_gate_pass": bool(r.fk_recheck_p95_m is not None and r.fk_recheck_p95_m <= float(args.fk_tol_m)),
        }
        for r in audits
    ]
    payload = {
        "mode": "large_ellipse_data_audit",
        "robot_config": str(args.robot_config),
        "sample_rows": int(args.sample_rows),
        "fk_tol_m": float(args.fk_tol_m),
        "datasets": [asdict(r) for r in audits],
        "fk_recheck": fk_rows,
    }
    (out_dir / "fk_recheck_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit large-ellipse optimization input datasets.")
    ap.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "00_data_audit")
    ap.add_argument("--sample-rows", type=int, default=1000)
    ap.add_argument("--fk-tol-m", type=float, default=1.0e-6)
    ap.add_argument("--seed", type=int, default=20260707)
    return ap.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({"out_dir": "runs/dataset_optimization_large_ellipse_v1/00_data_audit", "datasets": len(payload["datasets"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
