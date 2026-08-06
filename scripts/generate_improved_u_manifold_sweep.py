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
from sklearn.neighbors import NearestNeighbors

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "runs" / ".mplconfig"))

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
XYZ_COLS = ["x_m", "y_m", "z_m"]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    s10: float
    s20: float
    ds1: float
    ds2: float
    omega1_deg: float = 0.0
    omega2_deg: float = 0.0
    dq: float = 0.0
    base_id: str = ""


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def candidate_specs(families: set[str]) -> list[CandidateSpec]:
    family_a = {
        "A1": (0.125, 0.250, +0.075, +0.100),
        "A2": (0.125, 0.250, +0.100, -0.100),
        "A3": (0.125, 0.250, -0.100, +0.150),
        "A4": (0.150, 0.300, +0.125, -0.150),
        "A5": (0.100, 0.350, +0.150, -0.200),
        "A6": (0.175, 0.225, -0.125, +0.175),
    }
    out: list[CandidateSpec] = []
    if "A" in families:
        for cid, vals in family_a.items():
            out.append(CandidateSpec(cid, "A", *vals, base_id=cid))
    family_b = {
        "B1": ("A2", +5.0, -5.0),
        "B2": ("A2", +10.0, -10.0),
        "B3": ("A3", -5.0, +5.0),
        "B4": ("A4", +8.0, -8.0),
        "B5": ("A5", +10.0, -5.0),
    }
    b_specs: list[CandidateSpec] = []
    if "B" in families:
        for cid, (base, w1, w2) in family_b.items():
            b_specs.append(CandidateSpec(cid, "B", *family_a[base], omega1_deg=w1, omega2_deg=w2, base_id=base))
        out.extend(b_specs)
    if "C" in families:
        base_for_c: list[CandidateSpec] = []
        for base in ("A2", "A3", "A4", "A5", "A6"):
            base_for_c.append(CandidateSpec(base, "A", *family_a[base], base_id=base))
        for cid, (base, w1, w2) in family_b.items():
            base_for_c.append(CandidateSpec(cid, "B", *family_a[base], omega1_deg=w1, omega2_deg=w2, base_id=base))
        for base in base_for_c:
            for dq in (-0.15, -0.10, +0.10, +0.15):
                suffix = str(dq).replace("-", "m").replace("+", "p").replace(".", "p")
                out.append(
                    CandidateSpec(
                        f"C_{base.candidate_id}_dq{suffix}",
                        "C",
                        base.s10,
                        base.s20,
                        base.ds1,
                        base.ds2,
                        omega1_deg=base.omega1_deg,
                        omega2_deg=base.omega2_deg,
                        dq=float(dq),
                        base_id=base.candidate_id,
                    )
                )
    return out


def range_values(raw: str) -> np.ndarray:
    parts = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("range must be lo,hi,step")
    lo, hi, step = parts
    n = int(round((hi - lo) / step)) + 1
    vals = lo + np.arange(n, dtype=float) * step
    vals[-1] = hi
    return vals


def build_u_grid(a_deg: np.ndarray, b_deg: np.ndarray, eta_count: int) -> pd.DataFrame:
    eta = np.linspace(-1.0, 1.0, int(eta_count), dtype=float)
    aa, bb, ee = np.meshgrid(a_deg, b_deg, eta, indexing="ij")
    out = pd.DataFrame({"u_a_deg": aa.ravel(), "u_b_deg": bb.ravel(), "u_eta": ee.ravel()})
    out["u_a_rad"] = np.deg2rad(out["u_a_deg"].to_numpy(dtype=float))
    out["u_b_rad"] = np.deg2rad(out["u_b_deg"].to_numpy(dtype=float))
    out["rho_deg"] = np.sqrt(np.square(out["u_a_deg"]) + np.square(out["u_b_deg"]))
    return out


def beta_from_arrays(a: np.ndarray, b: np.ndarray, eta: np.ndarray, spec: CandidateSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    eta = np.asarray(eta, dtype=float).reshape(-1)
    s1 = float(spec.s10) + float(spec.ds1) * eta
    s2 = float(spec.s20) + float(spec.ds2) * eta
    q = 1.0 + float(spec.dq) * eta
    d = np.column_stack([a, b])

    def rotate(vec: np.ndarray, phi: np.ndarray) -> np.ndarray:
        c = np.cos(phi)
        s = np.sin(phi)
        return np.column_stack([c * vec[:, 0] - s * vec[:, 1], s * vec[:, 0] + c * vec[:, 1]])

    d1 = rotate(d, np.deg2rad(float(spec.omega1_deg)) * eta)
    d2 = rotate(d, np.deg2rad(float(spec.omega2_deg)) * eta)
    beta = np.column_stack([s1 * d1[:, 0], s1 * d1[:, 1], s2 * d2[:, 0], s2 * d2[:, 1], q * d[:, 0], q * d[:, 1]])
    return beta, s1, s2, q


def theta_from_beta_batch(beta: np.ndarray, *, theta_sign: float) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).reshape(-1, 6)
    theta = np.empty((len(beta), 30), dtype=float)
    for section, (odd_col, even_col) in enumerate(((0, 1), (2, 3), (4, 5))):
        start = section * 10
        theta[:, start : start + 10 : 2] = beta[:, odd_col : odd_col + 1]
        theta[:, start + 1 : start + 10 : 2] = beta[:, even_col : even_col + 1]
    return theta * float(theta_sign)


def attach_fk(pool: pd.DataFrame, spec: CandidateSpec, *, lengths_m: np.ndarray, p_end_local_m: np.ndarray, theta_sign: float) -> pd.DataFrame:
    beta, s1, s2, q = beta_from_arrays(
        pool["u_a_rad"].to_numpy(dtype=float),
        pool["u_b_rad"].to_numpy(dtype=float),
        pool["u_eta"].to_numpy(dtype=float),
        spec,
    )
    out = pool.copy().reset_index(drop=True)
    out["candidate_id"] = spec.candidate_id
    out["family"] = spec.family
    out["base_id"] = spec.base_id
    out["path_name"] = spec.candidate_id
    out["s1"] = s1
    out["s2"] = s2
    out["q_distal"] = q
    for i, col in enumerate(BETA_COLS):
        out[col] = beta[:, i]
    theta = theta_from_beta_batch(beta, theta_sign=theta_sign)
    xyz = fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    for i, col in enumerate(THETA_COLS):
        out[col] = theta[:, i]
    for i, col in enumerate(XYZ_COLS):
        out[col] = xyz[:, i]
    out["source_component"] = spec.candidate_id
    out["policy_version"] = "improved_u_manifold_sweep_v1"
    out["sample_id"] = np.arange(len(out), dtype=np.int64)
    return out


def attach_jacobian(
    df: pd.DataFrame,
    spec: CandidateSpec,
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    a_scale_rad: float,
    b_scale_rad: float,
    delta: float,
    sigma3_min_m: float,
    kappa_max: float,
    batch_size: int,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    u_bar = np.column_stack(
        [
            out["u_a_rad"].to_numpy(dtype=float) / float(a_scale_rad),
            out["u_b_rad"].to_numpy(dtype=float) / float(b_scale_rad),
            out["u_eta"].to_numpy(dtype=float),
        ]
    )
    singular = np.empty((len(out), 3), dtype=float)
    for start in range(0, len(out), int(batch_size)):
        end = min(len(out), start + int(batch_size))
        base = u_bar[start:end]
        cols = []
        for j in range(3):
            step = np.zeros_like(base)
            step[:, j] = float(delta)
            plus = base + step
            minus = base - step
            for arr in (plus, minus):
                beta, _s1, _s2, _q = beta_from_arrays(arr[:, 0] * a_scale_rad, arr[:, 1] * b_scale_rad, arr[:, 2], spec)
                theta = theta_from_beta_batch(beta, theta_sign=theta_sign)
                xyz = fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
                cols.append(xyz)
        jac = np.empty((end - start, 3, 3), dtype=float)
        for j in range(3):
            xyz_p = cols[2 * j]
            xyz_m = cols[2 * j + 1]
            jac[:, :, j] = (xyz_p - xyz_m) / (2.0 * float(delta))
        singular[start:end, :] = np.linalg.svd(jac, compute_uv=False)
    out["sigma1_m"] = singular[:, 0]
    out["sigma2_m"] = singular[:, 1]
    out["sigma3_m"] = singular[:, 2]
    out["kappa"] = singular[:, 0] / np.maximum(singular[:, 2], 1.0e-12)
    out["jacobian_gate_pass"] = (out["sigma3_m"] >= float(sigma3_min_m)) & (out["kappa"] <= float(kappa_max))
    return out


def local_beta_rms_p95(df: pd.DataFrame, *, radius_mm: float = 10.0, k: int = 16, max_rows: int = 12000, seed: int = 20260707) -> float:
    if len(df) < 2:
        return float("nan")
    work = df
    if len(work) > int(max_rows):
        work = work.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)
    xyz = work[XYZ_COLS].to_numpy(dtype=float)
    beta = work[BETA_COLS].to_numpy(dtype=float)
    kk = min(max(2, int(k)), len(work))
    nn = NearestNeighbors(n_neighbors=kk).fit(xyz)
    dist, idx = nn.kneighbors(xyz)
    vals: list[float] = []
    radius_m = float(radius_mm) / 1000.0
    for i in range(len(work)):
        mask = (dist[i] > 1.0e-12) & (dist[i] <= radius_m)
        if not np.any(mask):
            continue
        diff = beta[idx[i][mask]] - beta[i]
        rms = np.sqrt(np.mean(np.square(diff), axis=1)) * (180.0 / math.pi)
        vals.extend(float(v) for v in rms)
    return float(np.percentile(vals, 95)) if vals else float("nan")


def quick_supported_amp(df: pd.DataFrame, *, seed: int, max_centers: int, n_points: int) -> dict[str, Any]:
    if len(df) == 0:
        return {"max_support_amp_xy_mm": None, "best_nn_p95_mm": None, "best_x_bias_mm": None}
    xyz = df[XYZ_COLS].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=1).fit(xyz)
    rng = np.random.default_rng(int(seed))
    center_idx = rng.choice(len(df), size=min(int(max_centers), len(df)), replace=False)
    centers = xyz[center_idx]
    amps = [25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0]
    phases = [(0.0, 0.0, 0.0), (math.pi / 2, math.pi / 2, math.pi / 2), (math.pi, math.pi, math.pi), (3 * math.pi / 2, 3 * math.pi / 2, 3 * math.pi / 2)]
    best: dict[str, Any] = {"max_support_amp_xy_mm": None, "best_nn_p95_mm": None, "best_x_bias_mm": None}
    t = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False)
    for center in centers:
        for amp in amps:
            for px, py, pz in phases:
                target = np.empty((len(t), 3), dtype=float)
                target[:, 0] = center[0] + (amp / 1000.0) * np.sin(t + px)
                target[:, 1] = center[1] + (amp / 1000.0) * np.sin(t + py)
                target[:, 2] = center[2] + (1.5 * amp / 1000.0) * np.sin(t + pz)
                dist, idx = nn.kneighbors(target)
                nearest = xyz[idx[:, 0]]
                nn_p95 = float(np.percentile(dist[:, 0] * 1000.0, 95))
                x_bias = float(np.mean((nearest[:, 0] - target[:, 0]) * 1000.0))
                if nn_p95 <= 8.0 and abs(x_bias) <= 2.0:
                    if best["max_support_amp_xy_mm"] is None or amp > float(best["max_support_amp_xy_mm"]):
                        best = {"max_support_amp_xy_mm": amp, "best_nn_p95_mm": nn_p95, "best_x_bias_mm": x_bias}
    return best


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def run_candidate(spec: CandidateSpec, u_grid: pd.DataFrame, args: argparse.Namespace, *, lengths_m: np.ndarray, p_end_local_m: np.ndarray, theta_sign: float) -> dict[str, Any]:
    cand_dir = Path(args.out_dir) / spec.candidate_id
    full = attach_fk(u_grid, spec, lengths_m=lengths_m, p_end_local_m=p_end_local_m, theta_sign=theta_sign)
    keep_bounds = (
        (full["s1"] >= float(args.s1_min))
        & (full["s1"] <= float(args.s1_max))
        & (full["s2"] >= float(args.s2_min))
        & (full["s2"] <= float(args.s2_max))
        & (full["q_distal"] > 0.0)
    )
    full = full.loc[keep_bounds].copy().reset_index(drop=True)
    x_pool = full[(full["x_m"] >= float(args.x_min)) & (full["x_m"] <= float(args.x_max)) & (full["rho_deg"] >= float(args.rho_min_deg))].copy().reset_index(drop=True)
    jac = attach_jacobian(
        x_pool,
        spec,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        a_scale_rad=float(np.deg2rad(max(abs(float(args.a_deg.split(',')[0])), abs(float(args.a_deg.split(',')[1]))))),
        b_scale_rad=float(np.deg2rad(max(abs(float(args.b_deg.split(',')[0])), abs(float(args.b_deg.split(',')[1]))))),
        delta=float(args.jacobian_delta),
        sigma3_min_m=float(args.sigma3_min_m),
        kappa_max=float(args.kappa_max),
        batch_size=int(args.jacobian_batch_size),
    )
    jac_pass = jac[jac["jacobian_gate_pass"].astype(bool)].copy().reset_index(drop=True)

    if bool(args.write_pools):
        write_parquet(full, cand_dir / "full_pool.parquet")
        write_parquet(x_pool, cand_dir / "x1p0_1p2_pool.parquet")
        write_parquet(jac_pass, cand_dir / "jacobian_pass_pool.parquet")

    eval_df = jac_pass if len(jac_pass) else x_pool
    xyz = eval_df[XYZ_COLS].to_numpy(dtype=float) if len(eval_df) else np.empty((0, 3), dtype=float)
    rho = np.sqrt(np.square(xyz[:, 1]) + np.square(xyz[:, 2])) if len(xyz) else np.asarray([])
    all10 = local_beta_rms_p95(eval_df, seed=int(args.seed)) if len(eval_df) else float("nan")
    support = quick_supported_amp(eval_df, seed=int(args.seed), max_centers=int(args.ellipse_centers), n_points=int(args.ellipse_points))
    report = {
        **asdict(spec),
        "full_rows": int(len(full)),
        "x_rows": int(len(x_pool)),
        "jac_pass_rows": int(len(jac_pass)),
        "jac_pass_ratio": float(len(jac_pass) / max(1, len(x_pool))),
        "x_min": float(np.min(xyz[:, 0])) if len(xyz) else float("nan"),
        "x_max": float(np.max(xyz[:, 0])) if len(xyz) else float("nan"),
        "rho_min": float(np.min(rho)) if len(rho) else float("nan"),
        "rho_max": float(np.max(rho)) if len(rho) else float("nan"),
        "sigma3_p50_m": float(np.percentile(jac["sigma3_m"], 50)) if len(jac) else float("nan"),
        "sigma3_p95_m": float(np.percentile(jac["sigma3_m"], 95)) if len(jac) else float("nan"),
        "kappa_p50": float(np.percentile(jac["kappa"], 50)) if len(jac) else float("nan"),
        "kappa_p95": float(np.percentile(jac["kappa"], 95)) if len(jac) else float("nan"),
        "all10_beta_rms_p95_deg": all10,
        **support,
    }
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return report


def write_top_candidates(out_dir: Path, summary: pd.DataFrame) -> None:
    cols = [
        "candidate_id",
        "family",
        "base_id",
        "dq",
        "omega1_deg",
        "omega2_deg",
        "x_rows",
        "jac_pass_rows",
        "kappa_p95",
        "all10_beta_rms_p95_deg",
        "max_support_amp_xy_mm",
        "best_nn_p95_mm",
    ]
    lines = ["# Improved u-manifold sweep top candidates", ""]
    ranked = summary.sort_values(["max_support_amp_xy_mm", "all10_beta_rms_p95_deg", "kappa_p95"], ascending=[False, True, True])
    lines += [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in ranked.head(20).itertuples(index=False):
        d = row._asdict()
        vals = []
        for col in cols:
            value = d.get(col)
            if isinstance(value, float):
                vals.append("" if not math.isfinite(value) else f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    (out_dir / "top_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.robot_config))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    u_grid = build_u_grid(range_values(args.a_deg), range_values(args.b_deg), int(args.eta_count))
    families = {v.strip().upper() for v in str(args.families).split(",") if v.strip()}
    specs = candidate_specs(families)
    if args.only:
        wanted = {v.strip() for v in str(args.only).split(",") if v.strip()}
        specs = [s for s in specs if s.candidate_id in wanted]
    if int(args.max_candidates) > 0:
        specs = specs[: int(args.max_candidates)]
    rows = []
    for i, spec in enumerate(specs):
        if bool(args.progress):
            print(f"[{i + 1}/{len(specs)}] {spec.candidate_id}", flush=True)
        rows.append(run_candidate(spec, u_grid, args, lengths_m=inputs.lengths_m, p_end_local_m=inputs.p_end_local_m, theta_sign=theta_sign))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "candidate_summary.csv", index=False)
    write_top_candidates(out_dir, summary)
    payload = {"mode": "improved_u_manifold_sweep", "out_dir": str(out_dir), "candidates": rows}
    (out_dir / "sweep_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate and diagnose improved canonical u-manifold candidates.")
    ap.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "dataset_optimization_large_ellipse_v1" / "03_improved_u_manifold_sweep")
    ap.add_argument("--families", default="A,B,C")
    ap.add_argument("--only", default="")
    ap.add_argument("--a-deg", default="-15,15,1")
    ap.add_argument("--b-deg", default="-15,15,1")
    ap.add_argument("--eta-count", type=int, default=21)
    ap.add_argument("--x-min", type=float, default=1.0)
    ap.add_argument("--x-max", type=float, default=1.2)
    ap.add_argument("--rho-min-deg", type=float, default=2.0)
    ap.add_argument("--s1-min", type=float, default=0.0)
    ap.add_argument("--s1-max", type=float, default=0.35)
    ap.add_argument("--s2-min", type=float, default=0.05)
    ap.add_argument("--s2-max", type=float, default=0.60)
    ap.add_argument("--sigma3-min-m", type=float, default=0.002)
    ap.add_argument("--kappa-max", type=float, default=50.0)
    ap.add_argument("--jacobian-delta", type=float, default=1.0e-3)
    ap.add_argument("--jacobian-batch-size", type=int, default=4096)
    ap.add_argument("--ellipse-centers", type=int, default=150)
    ap.add_argument("--ellipse-points", type=int, default=180)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--write-pools", action="store_true")
    ap.add_argument("--progress", action="store_true")
    return ap.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({"out_dir": payload["out_dir"], "candidates": len(payload["candidates"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
