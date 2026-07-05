#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]


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
    keep = ["sample_id"] + [c for c in BETA_COLS + ["source_component"] if c in meta.columns]
    df = dataset.merge(meta[keep], on="sample_id", how="left", validate="one_to_one")
    if "source_component" not in df.columns:
        df["source_component"] = "unknown"
    df["source_component"] = df["source_component"].fillna("unknown").astype(str)
    if df[BETA_COLS].isna().any().any():
        raise ValueError("meta is missing beta columns for at least one row")
    return df


def _normalized_beta(beta: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.nanmax(np.abs(beta), axis=0), 1.0e-9)
    return beta / scale.reshape(1, -1)


def _radius_pairs(xyz: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) < 2:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    nn = NearestNeighbors(radius=float(radius_m), algorithm="auto").fit(xyz)
    neigh = nn.radius_neighbors(xyz, return_distance=False)
    pairs: set[tuple[int, int]] = set()
    for row, vals in enumerate(neigh):
        for col in vals.tolist():
            col = int(col)
            if col == row:
                continue
            a, b = (row, col) if row < col else (col, row)
            pairs.add((a, b))
    if not pairs:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    arr = np.asarray(sorted(pairs), dtype=int)
    return arr[:, 0], arr[:, 1]


def _metric_block(
    *,
    i: np.ndarray,
    j: np.ndarray,
    xyz: np.ndarray,
    beta_norm: np.ndarray,
    theta: np.ndarray,
    tension: np.ndarray,
    beta_close_threshold_norm: float,
) -> dict[str, Any]:
    i = np.asarray(i, dtype=int).reshape(-1)
    j = np.asarray(j, dtype=int).reshape(-1)
    if i.size == 0:
        return {"pairs": 0, "beta_far_ratio": 0.0}
    dxyz_mm = np.linalg.norm(xyz[i] - xyz[j], axis=1) * 1000.0
    dbeta = np.sqrt(np.mean(np.square(beta_norm[i] - beta_norm[j]), axis=1))
    theta_rms_deg = np.sqrt(np.mean(np.square(theta[i] - theta[j]), axis=1)) * (180.0 / np.pi)
    tension_mae = np.mean(np.abs(tension[i] - tension[j]), axis=1)
    beta_far = dbeta > float(beta_close_threshold_norm)
    return {
        "pairs": int(i.size),
        "xyz_dist_mm_p50": float(np.percentile(dxyz_mm, 50)),
        "xyz_dist_mm_p95": float(np.percentile(dxyz_mm, 95)),
        "beta_dist_norm_p50": float(np.percentile(dbeta, 50)),
        "beta_dist_norm_p95": float(np.percentile(dbeta, 95)),
        "beta_close_pairs": int(np.sum(~beta_far)),
        "beta_far_pairs": int(np.sum(beta_far)),
        "beta_far_ratio": float(np.mean(beta_far)),
        "theta_rms_deg_p50": float(np.percentile(theta_rms_deg, 50)),
        "theta_rms_deg_p95": float(np.percentile(theta_rms_deg, 95)),
        "tension_mae_n_p50": float(np.percentile(tension_mae, 50)),
        "tension_mae_n_p90": float(np.percentile(tension_mae, 90)),
        "tension_mae_n_p95": float(np.percentile(tension_mae, 95)),
    }


def _source_pair_key(a: str, b: str) -> str:
    left, right = sorted([str(a), str(b)])
    return f"{left}|{right}"


def _risk_score(within: dict[str, Any], cross_blocks: list[dict[str, Any]]) -> float:
    score = 0.0
    for block in cross_blocks:
        pairs = float(block.get("pairs", 0))
        score += pairs * float(block.get("beta_far_ratio", 0.0)) * (
            1.0
            + float(block.get("tension_mae_n_p95", 0.0)) / 100.0
            + float(block.get("theta_rms_deg_p95", 0.0)) / 5.0
        )
    score += 0.25 * float(within.get("pairs", 0)) * float(within.get("beta_far_ratio", 0.0))
    score += 0.01 * float(within.get("tension_mae_n_p95", 0.0))
    return float(score)


def evaluate_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    radius_m: float = 0.01,
    beta_close_threshold_norm: float = 0.15,
) -> dict[str, Any]:
    df = _merge(dataset, meta)
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    beta_norm = _normalized_beta(df[BETA_COLS].to_numpy(dtype=float))
    theta_cols = _numbered_cols(list(df.columns), "theta_", "_rad")
    tension_cols = _numbered_cols(list(df.columns), "tension_", "_n")
    if len(theta_cols) != 30 or len(tension_cols) != 12:
        raise ValueError("dataset must contain 30 theta columns and 12 tension columns")
    theta = df[theta_cols].to_numpy(dtype=float)
    tension = df[tension_cols].to_numpy(dtype=float)
    sources = df["source_component"].astype(str).to_numpy()
    source_names = sorted(df["source_component"].astype(str).unique().tolist())
    pi, pj = _radius_pairs(xyz, radius_m)

    source_payload: dict[str, Any] = {}
    pair_matrix: dict[str, Any] = {}
    cross_by_source: dict[str, list[dict[str, Any]]] = {src: [] for src in source_names}

    for src in source_names:
        m = (sources[pi] == src) & (sources[pj] == src)
        within = _metric_block(
            i=pi[m],
            j=pj[m],
            xyz=xyz,
            beta_norm=beta_norm,
            theta=theta,
            tension=tension,
            beta_close_threshold_norm=beta_close_threshold_norm,
        )
        source_payload[src] = {"rows": int(np.sum(sources == src)), "within_source": within}

    for a_idx, a in enumerate(source_names):
        for b in source_names[a_idx + 1 :]:
            m = ((sources[pi] == a) & (sources[pj] == b)) | ((sources[pi] == b) & (sources[pj] == a))
            block = _metric_block(
                i=pi[m],
                j=pj[m],
                xyz=xyz,
                beta_norm=beta_norm,
                theta=theta,
                tension=tension,
                beta_close_threshold_norm=beta_close_threshold_norm,
            )
            key = _source_pair_key(a, b)
            pair_matrix[key] = block
            cross_by_source[a].append(block)
            cross_by_source[b].append(block)

    ranking = []
    for src in source_names:
        ranking.append(
            {
                "source": src,
                "rows": source_payload[src]["rows"],
                "risk_score": _risk_score(source_payload[src]["within_source"], cross_by_source[src]),
                "within_tension_p95_n": float(source_payload[src]["within_source"].get("tension_mae_n_p95", 0.0)),
                "cross_pair_count": int(sum(int(b.get("pairs", 0)) for b in cross_by_source[src])),
                "cross_beta_far_pairs": int(sum(int(b.get("beta_far_pairs", 0)) for b in cross_by_source[src])),
            }
        )
    ranking.sort(key=lambda r: (-float(r["risk_score"]), str(r["source"])))
    return {
        "rows": int(len(df)),
        "radius_m": float(radius_m),
        "beta_close_threshold_norm": float(beta_close_threshold_norm),
        "sources": source_payload,
        "source_pair_matrix": pair_matrix,
        "source_risk_ranking": ranking,
    }


def evaluate(dataset_path: Path, meta_path: Path, **kwargs: Any) -> dict[str, Any]:
    return evaluate_frames(pd.read_parquet(dataset_path), pd.read_parquet(meta_path), **kwargs)


def _write_report(payload: dict[str, Any], out_path: Path) -> None:
    def fmt(v: Any, digits: int = 2) -> str:
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return str(v)

    lines = [
        "# Source Ablation Report",
        "",
        f"- rows: {payload['rows']}",
        f"- radius: {float(payload['radius_m']) * 1000.0:.1f} mm",
        f"- beta_close_threshold_norm: {payload['beta_close_threshold_norm']}",
        "",
        "## Source Risk Ranking",
        "",
        "| rank | source | rows | risk score | cross pairs | cross beta-far pairs | within T p95 N |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(payload["source_risk_ranking"], start=1):
        lines.append(
            f"| {rank} | {row['source']} | {row['rows']} | {fmt(row['risk_score'], 3)} | "
            f"{row['cross_pair_count']} | {row['cross_beta_far_pairs']} | {fmt(row['within_tension_p95_n'])} |"
        )
    lines.extend(
        [
            "",
            "## Source Pair Matrix",
            "",
            "| source pair | pairs | beta-far ratio | T p50 N | T p95 N | theta p95 deg |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, row in sorted(payload["source_pair_matrix"].items()):
        lines.append(
            f"| {key} | {row.get('pairs', 0)} | {fmt(row.get('beta_far_ratio', 0.0), 3)} | "
            f"{fmt(row.get('tension_mae_n_p50', 0.0))} | {fmt(row.get('tension_mae_n_p95', 0.0))} | "
            f"{fmt(row.get('theta_rms_deg_p95', 0.0), 3)} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--radius-mm", type=float, default=10.0)
    ap.add_argument("--beta-close-threshold-norm", type=float, default=0.15)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = evaluate(
        args.dataset,
        args.meta,
        radius_m=float(args.radius_mm) / 1000.0,
        beta_close_threshold_norm=float(args.beta_close_threshold_norm),
    )
    (args.out_dir / "source_ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(payload, args.out_dir / "source_ablation_report.md")
    print(json.dumps({"out_dir": str(args.out_dir), "top_source": payload["source_risk_ranking"][0]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

