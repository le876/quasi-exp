from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _theta_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("theta_") and c.endswith("_rad")]


def _tension_cols(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if not (c.startswith("tension_") and c.endswith("_n")):
            continue
        raw = c[len("tension_") : -len("_n")]
        if raw.isdigit():
            out.append((int(raw), c))
    return [c for _, c in sorted(out)]


def _theta6_from_theta30(theta30: np.ndarray) -> np.ndarray:
    idx = [0, 1, 10, 11, 20, 21]
    return theta30[:, idx]


def evaluate(dataset_path: Path, radii_m: list[float], k_neighbors: int) -> dict:
    df = pd.read_parquet(dataset_path)
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    theta30 = df[_theta_cols(df)].to_numpy(dtype=float)
    theta6 = _theta6_from_theta30(theta30)
    tension_cols = _tension_cols(df)
    tension = df[tension_cols].to_numpy(dtype=float) if len(tension_cols) == 12 else None

    nn2 = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(xyz)
    d2, _ = nn2.kneighbors(xyz)
    nearest = d2[:, 1]

    k = min(max(2, int(k_neighbors)), len(df))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xyz)
    d, idx = nn.kneighbors(xyz)
    i = np.repeat(np.arange(len(df))[:, None], k - 1, axis=1).ravel()
    j = idx[:, 1:].ravel()
    dxyz = d[:, 1:].ravel()

    dtheta = theta6[i] - theta6[j]
    dtheta_rms_deg = np.sqrt(np.mean(np.square(dtheta), axis=1)) * (180.0 / np.pi)
    dtheta_max_deg = np.max(np.abs(dtheta), axis=1) * (180.0 / np.pi)
    if tension is not None:
        dtension = tension[i] - tension[j]
        dtension_abs = np.abs(dtension)
        dtension_mae_n = np.mean(dtension_abs, axis=1)
        dtension_max_abs_n = np.max(dtension_abs, axis=1)
    else:
        dtension_mae_n = None
        dtension_max_abs_n = None

    result: dict[str, object] = {
        "dataset": str(dataset_path),
        "rows": int(len(df)),
        "theta_independent_dims": int(theta6.shape[1]),
        "tension_dims": int(0 if tension is None else tension.shape[1]),
        "nn": {
            "dxyz_mm_min": float(np.min(nearest) * 1000.0),
            "dxyz_mm_mean": float(np.mean(nearest) * 1000.0),
            "dxyz_mm_p50": float(np.percentile(nearest, 50) * 1000.0),
            "dxyz_mm_p90": float(np.percentile(nearest, 90) * 1000.0),
            "dxyz_mm_p95": float(np.percentile(nearest, 95) * 1000.0),
        },
        "local": {},
    }

    for r in radii_m:
        key = f"<= {int(r * 1000)}mm"
        m = dxyz <= r
        if not np.any(m):
            result["local"][key] = {"pairs": 0}
            continue
        rr = dtheta_rms_deg[m]
        rm = dtheta_max_deg[m]
        block = {
            "pairs": int(np.sum(m)),
            "theta_rms_deg_mean": float(np.mean(rr)),
            "theta_rms_deg_p90": float(np.percentile(rr, 90)),
            "theta_rms_deg_p95": float(np.percentile(rr, 95)),
            "theta_max_deg_p95": float(np.percentile(rm, 95)),
            "ratio_rms_gt1deg": float(np.mean(rr > 1.0)),
            "ratio_rms_gt2deg": float(np.mean(rr > 2.0)),
            "ratio_rms_gt5deg": float(np.mean(rr > 5.0)),
            "ratio_rms_gt10deg": float(np.mean(rr > 10.0)),
        }
        if dtension_mae_n is not None and dtension_max_abs_n is not None:
            tm = dtension_mae_n[m]
            tx = dtension_max_abs_n[m]
            block.update(
                {
                    "tension_mae_n_p50": float(np.percentile(tm, 50)),
                    "tension_mae_n_p90": float(np.percentile(tm, 90)),
                    "tension_mae_n_p95": float(np.percentile(tm, 95)),
                    "tension_mae_n_max": float(np.max(tm)),
                    "tension_max_abs_n_p95": float(np.percentile(tx, 95)),
                    "tension_max_abs_n_max": float(np.max(tx)),
                }
            )
        result["local"][key] = block

    return result


def gate(result: dict) -> tuple[bool, dict]:
    local = result["local"]
    m10 = local.get("<= 10mm", {})
    m20 = local.get("<= 20mm", {})

    checks = {
        "pairs_10mm_ge_100": bool(m10.get("pairs", 0) >= 100),
        "pairs_20mm_ge_500": bool(m20.get("pairs", 0) >= 500),
        "p95_10mm_le_6deg": bool(float(m10.get("theta_rms_deg_p95", 1e9)) <= 6.0),
        "p95_20mm_le_6deg": bool(float(m20.get("theta_rms_deg_p95", 1e9)) <= 6.0),
        "gt10_10mm_le_1pct": bool(float(m10.get("ratio_rms_gt10deg", 1.0)) <= 0.01),
        "gt10_20mm_le_2pct": bool(float(m20.get("ratio_rms_gt10deg", 1.0)) <= 0.02),
    }
    if "tension_mae_n_p95" in m10:
        checks["tension_mae_10mm_p95_le_75n"] = bool(float(m10.get("tension_mae_n_p95", 1e9)) <= 75.0)
    passed = all(checks.values())
    return passed, checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--k-neighbors", type=int, default=80)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    res = evaluate(dataset_path, radii_m=[0.01, 0.02, 0.05, 0.10], k_neighbors=args.k_neighbors)
    passed, checks = gate(res)
    payload = {"passed": passed, "checks": checks, "metrics": res}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
