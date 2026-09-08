#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


TENSION_COLS = [f"tension_{i}_n" for i in range(1, 13)]
BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
EFFECTIVE_BETA_COLS = [f"effective_beta_{i}_rad" for i in range(1, 7)]
CASE_COLS = [f"case_{i}" for i in range(1, 13)]
XYZ_COLS = ["x_m", "y_m", "z_m"]
SEGMENT_CABLES = {
    "first": np.asarray([0, 1, 10, 11], dtype=int),
    "second": np.asarray([2, 3, 8, 9], dtype=int),
    "third": np.asarray([4, 5, 6, 7], dtype=int),
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _numbered_cols(columns: list[str], prefix: str, suffix: str) -> list[str]:
    out: list[tuple[int, str]] = []
    for col in columns:
        if not (col.startswith(prefix) and col.endswith(suffix)):
            continue
        raw = col[len(prefix) : -len(suffix)]
        if raw.isdigit():
            out.append((int(raw), col))
    return [col for _, col in sorted(out)]


def _merge(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in dataset.columns or "sample_id" not in meta.columns:
        raise ValueError("dataset and meta must contain sample_id")
    keep = ["sample_id"] + [c for c in EFFECTIVE_BETA_COLS + BETA_COLS + CASE_COLS + ["layer_label", "source_component"] if c in meta.columns]
    out = dataset.merge(meta[keep], on="sample_id", how="left", validate="one_to_one")
    for col in CASE_COLS:
        if col not in out.columns:
            out[col] = 0
    if "layer_label" not in out.columns:
        out["layer_label"] = out.get("source_component", "unknown")
    out["layer_label"] = out["layer_label"].fillna("unknown").astype(str)
    return out


def _beta_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = EFFECTIVE_BETA_COLS if set(EFFECTIVE_BETA_COLS).issubset(df.columns) else BETA_COLS
    if not set(cols).issubset(df.columns):
        raise ValueError("missing beta/effective_beta columns")
    beta = df[cols].to_numpy(dtype=float)
    scale = np.maximum(np.nanmax(np.abs(beta), axis=0), 1.0e-9)
    return beta / scale.reshape(1, 6)


def _radius_pairs(xyz: np.ndarray, radius_m: float, k_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) < 2:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    nn = NearestNeighbors(radius=float(radius_m), algorithm="auto").fit(xyz)
    neigh = nn.radius_neighbors(xyz, return_distance=False)
    pairs: set[tuple[int, int]] = set()
    for row, raw in enumerate(neigh):
        vals = [int(v) for v in raw.tolist() if int(v) != row]
        if int(k_neighbors) > 0 and len(vals) > int(k_neighbors):
            dist = np.linalg.norm(xyz[vals] - xyz[row], axis=1)
            vals = [vals[int(k)] for k in np.argsort(dist)[: int(k_neighbors)]]
        for col in vals:
            a, b = (row, col) if row < col else (col, row)
            pairs.add((a, b))
    if not pairs:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    arr = np.asarray(sorted(pairs), dtype=int)
    return arr[:, 0], arr[:, 1]


def _segment_causing_jump(tension: np.ndarray, i: int, j: int) -> str:
    scores = {
        name: float(np.mean(np.abs(tension[i, idx] - tension[j, idx])))
        for name, idx in SEGMENT_CABLES.items()
    }
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _active_signature(tension: np.ndarray) -> np.ndarray:
    lower = tension <= 1.0
    upper = tension >= 1999.0
    return np.concatenate([lower, upper], axis=1)


def _case_array(df: pd.DataFrame) -> np.ndarray:
    return df[CASE_COLS].fillna(0).to_numpy(dtype=int)


def _empty_group() -> dict[str, Any]:
    return {
        "pairs": 0,
        "tension_mae_n_p50": None,
        "tension_mae_n_p95": None,
        "tension_max_abs_n_p95": None,
        "case_flip_ratio": None,
        "active_set_flip_ratio": None,
        "segment_causing_jump_counts": {},
    }


def _group_metrics(
    *,
    i: np.ndarray,
    j: np.ndarray,
    tension: np.ndarray,
    case_flip: np.ndarray,
    active_flip: np.ndarray,
    segment_names: np.ndarray,
) -> dict[str, Any]:
    if i.size == 0:
        return _empty_group()
    t_mae = np.mean(np.abs(tension[i] - tension[j]), axis=1)
    t_max = np.max(np.abs(tension[i] - tension[j]), axis=1)
    counts = pd.Series(segment_names).value_counts().sort_index().to_dict()
    return {
        "pairs": int(i.size),
        "tension_mae_n_p50": float(np.percentile(t_mae, 50)),
        "tension_mae_n_p95": float(np.percentile(t_mae, 95)),
        "tension_max_abs_n_p95": float(np.percentile(t_max, 95)),
        "case_flip_ratio": float(np.mean(case_flip)) if case_flip.size else 0.0,
        "active_set_flip_ratio": float(np.mean(active_flip)) if active_flip.size else 0.0,
        "segment_causing_jump_counts": {str(k): int(v) for k, v in counts.items()},
    }


def audit_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    radius_m: float = 0.010,
    beta_close_threshold_norm: float = 0.15,
    k_neighbors: int = 80,
) -> dict[str, Any]:
    df = _merge(dataset, meta)
    theta_cols = _numbered_cols(list(df.columns), "theta_", "_rad")
    tension_cols = _numbered_cols(list(df.columns), "tension_", "_n")
    if len(theta_cols) != 30 or len(tension_cols) != 12:
        raise ValueError("dataset must contain 30 theta columns and 12 tension columns")
    xyz = df[XYZ_COLS].to_numpy(dtype=float)
    beta = _beta_matrix(df)
    tension = df[tension_cols].to_numpy(dtype=float)
    cases = _case_array(df)
    active = _active_signature(tension)

    i_all, j_all = _radius_pairs(xyz, float(radius_m), int(k_neighbors))
    if i_all.size:
        beta_dist = np.sqrt(np.mean(np.square(beta[i_all] - beta[j_all]), axis=1))
        keep = beta_dist <= float(beta_close_threshold_norm)
        i = i_all[keep]
        j = j_all[keep]
    else:
        i = i_all
        j = j_all

    case_flip = np.any(cases[i] != cases[j], axis=1) if i.size else np.asarray([], dtype=bool)
    active_flip = np.any(active[i] != active[j], axis=1) if i.size else np.asarray([], dtype=bool)
    segment_names = np.asarray([_segment_causing_jump(tension, int(a), int(b)) for a, b in zip(i.tolist(), j.tolist())], dtype=object)

    masks = {
        "no_flip": (~case_flip) & (~active_flip),
        "case_flip_only": case_flip & (~active_flip),
        "active_set_flip_only": (~case_flip) & active_flip,
        "both_flip": case_flip & active_flip,
        "all": np.ones(i.size, dtype=bool),
    }
    groups = {
        name: _group_metrics(
            i=i[mask],
            j=j[mask],
            tension=tension,
            case_flip=case_flip[mask],
            active_flip=active_flip[mask],
            segment_names=segment_names[mask],
        )
        for name, mask in masks.items()
    }
    return {
        "rows": int(len(df)),
        "pair_count": int(i.size),
        "radius_m": float(radius_m),
        "beta_close_threshold_norm": float(beta_close_threshold_norm),
        "k_neighbors": int(k_neighbors),
        "groups": groups,
    }


def audit(dataset_path: Path, meta_path: Path, **kwargs: Any) -> dict[str, Any]:
    return audit_frames(pd.read_parquet(dataset_path), pd.read_parquet(meta_path), **kwargs)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Tension Discontinuity Audit",
        "",
        f"- rows: {payload['rows']}",
        f"- beta-close pair count: {payload['pair_count']}",
        f"- radius_m: {payload['radius_m']}",
        "",
        "| group | pairs | T MAE p95 N | case flip ratio | active-set flip ratio | top segment |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, block in payload["groups"].items():
        counts = block.get("segment_causing_jump_counts") or {}
        top = "none"
        if counts:
            top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        p95 = block.get("tension_mae_n_p95")
        case_ratio = block.get("case_flip_ratio")
        active_ratio = block.get("active_set_flip_ratio")
        lines.append(
            f"| {name} | {block.get('pairs', 0)} | "
            f"{'nan' if p95 is None else f'{float(p95):.3f}'} | "
            f"{'nan' if case_ratio is None else f'{float(case_ratio):.3f}'} | "
            f"{'nan' if active_ratio is None else f'{float(active_ratio):.3f}'} | "
            f"{top} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--radius-mm", type=float, default=10.0)
    ap.add_argument("--beta-close-threshold-norm", type=float, default=0.15)
    ap.add_argument("--k-neighbors", type=int, default=80)
    args = ap.parse_args()
    payload = audit(
        args.dataset,
        args.meta,
        radius_m=float(args.radius_mm) / 1000.0,
        beta_close_threshold_norm=float(args.beta_close_threshold_norm),
        k_neighbors=int(args.k_neighbors),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "audit_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (args.out_dir / "REPORT.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
