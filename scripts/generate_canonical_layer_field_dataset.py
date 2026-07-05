#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))

from fk_dh_numpy import fk_dh_batch  # noqa: E402
from generate_hierarchical_beta_fk_dataset import (  # noqa: E402
    workspace_metrics,
    select_workspace_balanced_subset,
)
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
XYZ_COLS = ["x_m", "y_m", "z_m"]
POLICY_VERSION = "canonical_layer_field_u3_v1"


@dataclass(frozen=True)
class LayerPath:
    name: str
    s10: float
    s20: float
    ds1: float
    ds2: float

    def s1(self, eta: np.ndarray | float) -> np.ndarray:
        return float(self.s10) + float(self.ds1) * np.asarray(eta, dtype=float)

    def s2(self, eta: np.ndarray | float) -> np.ndarray:
        return float(self.s20) + float(self.ds2) * np.asarray(eta, dtype=float)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _load_analysis_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def layer_paths() -> dict[str, LayerPath]:
    return {
        "path_a_sync_plus": LayerPath("path_a_sync_plus", s10=0.125, s20=0.250, ds1=0.075, ds2=0.100),
        "path_b_redistribute_12": LayerPath("path_b_redistribute_12", s10=0.125, s20=0.250, ds1=0.075, ds2=-0.100),
        "path_c_redistribute_21": LayerPath("path_c_redistribute_21", s10=0.125, s20=0.250, ds1=-0.075, ds2=0.100),
        "path_d_wide_redistribute": LayerPath("path_d_wide_redistribute", s10=0.125, s20=0.250, ds1=0.125, ds2=-0.150),
    }


def _levels_from_range_deg(spec: tuple[float, float, float]) -> np.ndarray:
    lo, hi, step = (float(v) for v in spec)
    if not (lo < hi and step > 0.0):
        raise ValueError("range must satisfy lo < hi and step > 0")
    n = int(round((hi - lo) / step)) + 1
    vals = lo + np.arange(n, dtype=float) * step
    vals[-1] = hi
    return vals


def build_u_grid(*, a_deg: tuple[float, float, float], b_deg: tuple[float, float, float], eta_count: int) -> pd.DataFrame:
    if int(eta_count) < 2:
        raise ValueError("eta_count must be >= 2")
    a_vals = _levels_from_range_deg(a_deg)
    b_vals = _levels_from_range_deg(b_deg)
    eta_vals = np.linspace(-1.0, 1.0, int(eta_count), dtype=float)
    aa, bb, ee = np.meshgrid(a_vals, b_vals, eta_vals, indexing="ij")
    out = pd.DataFrame(
        {
            "u_a_deg": aa.reshape(-1),
            "u_b_deg": bb.reshape(-1),
            "u_eta": ee.reshape(-1),
        }
    )
    out["u_a_rad"] = np.deg2rad(out["u_a_deg"].to_numpy(dtype=float))
    out["u_b_rad"] = np.deg2rad(out["u_b_deg"].to_numpy(dtype=float))
    out["rho_deg"] = np.sqrt(np.square(out["u_a_deg"].to_numpy(dtype=float)) + np.square(out["u_b_deg"].to_numpy(dtype=float)))
    return out


def beta_from_u(a_rad: float, b_rad: float, eta: float, path: LayerPath) -> tuple[np.ndarray, float, float]:
    s1 = float(path.s1(float(eta)))
    s2 = float(path.s2(float(eta)))
    beta = np.asarray([s1 * a_rad, s1 * b_rad, s2 * a_rad, s2 * b_rad, a_rad, b_rad], dtype=float)
    return beta, s1, s2


def _beta_from_arrays(a_rad: np.ndarray, b_rad: np.ndarray, eta: np.ndarray, path: LayerPath) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(a_rad, dtype=float).reshape(-1)
    b = np.asarray(b_rad, dtype=float).reshape(-1)
    e = np.asarray(eta, dtype=float).reshape(-1)
    s1 = path.s1(e)
    s2 = path.s2(e)
    beta = np.column_stack([s1 * a, s1 * b, s2 * a, s2 * b, a, b])
    return beta, s1, s2


def theta_from_beta_batch(beta_rad: np.ndarray, *, theta_sign: float) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    theta = np.empty((len(beta), 30), dtype=float)
    for section, (odd_col, even_col) in enumerate(((0, 1), (2, 3), (4, 5))):
        start = section * 10
        theta[:, start : start + 10 : 2] = beta[:, odd_col : odd_col + 1]
        theta[:, start + 1 : start + 10 : 2] = beta[:, even_col : even_col + 1]
    return theta * float(theta_sign)


def build_layer_field_pool(
    u_grid: pd.DataFrame,
    path: LayerPath,
    *,
    s1_bounds: tuple[float, float] = (0.0, 0.35),
    s2_bounds: tuple[float, float] = (0.05, 0.60),
) -> pd.DataFrame:
    beta, s1, s2 = _beta_from_arrays(
        u_grid["u_a_rad"].to_numpy(dtype=float),
        u_grid["u_b_rad"].to_numpy(dtype=float),
        u_grid["u_eta"].to_numpy(dtype=float),
        path,
    )
    out = u_grid.copy().reset_index(drop=True)
    out["path_name"] = str(path.name)
    out["s1"] = s1
    out["s2"] = s2
    for i, col in enumerate(BETA_COLS):
        out[col] = beta[:, i]
    keep = (
        (out["s1"].to_numpy(dtype=float) >= float(s1_bounds[0]))
        & (out["s1"].to_numpy(dtype=float) <= float(s1_bounds[1]))
        & (out["s2"].to_numpy(dtype=float) >= float(s2_bounds[0]))
        & (out["s2"].to_numpy(dtype=float) <= float(s2_bounds[1]))
    )
    out = out.loc[keep].copy().reset_index(drop=True)
    out["source_component"] = str(path.name)
    out["policy_version"] = POLICY_VERSION
    out["sample_id"] = np.arange(len(out), dtype=np.int64)
    return out


def attach_fk(pool: pd.DataFrame, *, lengths_m: np.ndarray, p_end_local_m: np.ndarray, theta_sign: float) -> pd.DataFrame:
    out = pool.copy().reset_index(drop=True)
    beta = out[BETA_COLS].to_numpy(dtype=float)
    theta = theta_from_beta_batch(beta, theta_sign=float(theta_sign))
    xyz = fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    for i, col in enumerate(THETA_COLS):
        out[col] = theta[:, i]
    for i, col in enumerate(XYZ_COLS):
        out[col] = xyz[:, i]
    return out


def numerical_jacobian(
    h: Callable[[np.ndarray], np.ndarray],
    u_bar: np.ndarray,
    *,
    delta: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.asarray(u_bar, dtype=float).reshape(3)
    J = np.zeros((3, 3), dtype=float)
    for j in range(3):
        e = np.zeros(3, dtype=float)
        e[j] = 1.0
        xp = np.asarray(h(center + float(delta) * e), dtype=float).reshape(3)
        xm = np.asarray(h(center - float(delta) * e), dtype=float).reshape(3)
        J[:, j] = (xp - xm) / (2.0 * float(delta))
    singular = np.linalg.svd(J, compute_uv=False)
    kappa = float(singular[0] / max(float(singular[-1]), 1.0e-12))
    return J, singular, kappa


def _beta_from_u_bar_batch(u_bar: np.ndarray, *, path: LayerPath, a_scale_rad: float, b_scale_rad: float) -> np.ndarray:
    u = np.asarray(u_bar, dtype=float).reshape(-1, 3)
    a = float(a_scale_rad) * u[:, 0]
    b = float(b_scale_rad) * u[:, 1]
    eta = u[:, 2]
    beta, _s1, _s2 = _beta_from_arrays(a, b, eta, path)
    return beta


def attach_jacobian_gate(
    pool: pd.DataFrame,
    *,
    path: LayerPath,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    a_scale_rad: float,
    b_scale_rad: float,
    delta: float,
    sigma3_min_m: float,
    kappa_max: float,
) -> pd.DataFrame:
    out = pool.copy().reset_index(drop=True)
    n = len(out)
    if n == 0:
        for col in ["sigma1_m", "sigma2_m", "sigma3_m", "kappa"]:
            out[col] = pd.Series(dtype="float64")
        out["jacobian_gate_pass"] = pd.Series(dtype="bool")
        return out
    u_bar = np.column_stack(
        [
            out["u_a_rad"].to_numpy(dtype=float) / max(float(a_scale_rad), 1.0e-12),
            out["u_b_rad"].to_numpy(dtype=float) / max(float(b_scale_rad), 1.0e-12),
            out["u_eta"].to_numpy(dtype=float),
        ]
    )
    J = np.zeros((n, 3, 3), dtype=float)
    for j in range(3):
        e = np.zeros((1, 3), dtype=float)
        e[0, j] = 1.0
        beta_p = _beta_from_u_bar_batch(u_bar + float(delta) * e, path=path, a_scale_rad=a_scale_rad, b_scale_rad=b_scale_rad)
        beta_m = _beta_from_u_bar_batch(u_bar - float(delta) * e, path=path, a_scale_rad=a_scale_rad, b_scale_rad=b_scale_rad)
        theta_p = theta_from_beta_batch(beta_p, theta_sign=float(theta_sign))
        theta_m = theta_from_beta_batch(beta_m, theta_sign=float(theta_sign))
        xyz_p = fk_dh_batch(theta_p, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
        xyz_m = fk_dh_batch(theta_m, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
        J[:, :, j] = (xyz_p - xyz_m) / (2.0 * float(delta))
    singular = np.linalg.svd(J, compute_uv=False)
    sigma3 = singular[:, -1]
    kappa = singular[:, 0] / np.maximum(sigma3, 1.0e-12)
    out["sigma1_m"] = singular[:, 0]
    out["sigma2_m"] = singular[:, 1]
    out["sigma3_m"] = singular[:, 2]
    out["kappa"] = kappa
    out["jacobian_gate_pass"] = (sigma3 > float(sigma3_min_m)) & (kappa < float(kappa_max))
    return out


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def _plot_workspace(xyz: np.ndarray, out_path: Path, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if len(xyz) == 0:
        return
    if len(xyz) > 120000:
        rng = np.random.default_rng(20260705)
        xyz = xyz[np.sort(rng.choice(len(xyz), size=120000, replace=False))]
    views = [
        ("iso front", 24, -58),
        ("iso back", 24, 122),
        ("top xy", 90, -90),
        ("side xz", 0, -90),
        ("side yz", 0, 0),
        ("low oblique", 12, -35),
    ]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    norm = Normalize(vmin=float(xyz[:, 0].min()), vmax=float(xyz[:, 0].max()))
    fig = plt.figure(figsize=(18, 12))
    for idx, (name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 0], cmap="viridis", norm=norm, s=1.0, alpha=0.22, linewidths=0, rasterized=True)
        ax.set_xlim(float(mins[0]), float(maxs[0]))
        ax.set_ylim(float(mins[1]), float(maxs[1]))
        ax.set_zlim(float(mins[2]), float(maxs[2]))
        try:
            ax.set_box_aspect(tuple(np.maximum(maxs - mins, 1.0e-9).tolist()))
        except Exception:
            pass
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.64, pad=0.02)
    cbar.set_label("x position (m)")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_axis_hist(df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, XYZ_COLS):
        ax.hist(df[col].to_numpy(dtype=float), bins=80, color="#2f6f91", alpha=0.85)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _percentiles(values: np.ndarray, qs: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(q)}": float("nan") for q in qs}
    return {f"p{int(q)}": float(np.percentile(arr, float(q))) for q in qs}


def _dataset_and_meta(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy().reset_index(drop=True)
    work["sample_id"] = np.arange(len(work), dtype=np.int64)
    dataset_cols = ["sample_id", *XYZ_COLS, *THETA_COLS]
    meta_cols = [
        "sample_id",
        "path_name",
        "source_component",
        "policy_version",
        "u_a_deg",
        "u_b_deg",
        "u_eta",
        "s1",
        "s2",
        "rho_deg",
        *BETA_COLS,
        "sigma1_m",
        "sigma2_m",
        "sigma3_m",
        "kappa",
        "jacobian_gate_pass",
    ]
    dataset = work[[c for c in dataset_cols if c in work.columns]].copy()
    meta = work[[c for c in meta_cols if c in work.columns]].copy()
    return dataset, meta


def _local_branch_metrics(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    radius_m: float = 0.01,
    min_ball_size: int = 6,
    beta_eps_norm: float = 0.15,
) -> dict[str, Any]:
    if len(dataset) == 0:
        return {"rows": 0, "balls_evaluated": 0, "multi_branch_ball_ratio": 0.0}
    df = dataset[["sample_id", *XYZ_COLS]].merge(meta[["sample_id", *BETA_COLS]], on="sample_id", how="left", validate="one_to_one")
    xyz = df[XYZ_COLS].to_numpy(dtype=float)
    beta = df[BETA_COLS].to_numpy(dtype=float)
    scale = np.maximum(np.nanmax(np.abs(beta), axis=0), 1.0e-9)
    beta_norm = beta / scale.reshape(1, 6)
    nn = NearestNeighbors(radius=float(radius_m), algorithm="auto").fit(xyz)
    neigh = nn.radius_neighbors(xyz, return_distance=False)
    branch_counts: list[int] = []
    ball_sizes: list[int] = []
    for raw in neigh:
        idx = np.asarray(sorted(set(int(v) for v in raw.tolist())), dtype=int)
        if idx.size < int(min_ball_size):
            continue
        labels = DBSCAN(eps=float(beta_eps_norm), min_samples=1).fit_predict(beta_norm[idx])
        branch_counts.append(int(len(set(int(v) for v in labels.tolist()))))
        ball_sizes.append(int(idx.size))
    if not branch_counts:
        return {"rows": int(len(df)), "balls_evaluated": 0, "multi_branch_ball_ratio": 0.0}
    counts = np.asarray(branch_counts, dtype=float)
    return {
        "rows": int(len(df)),
        "radius_m": float(radius_m),
        "min_ball_size": int(min_ball_size),
        "beta_eps_norm": float(beta_eps_norm),
        "balls_evaluated": int(len(counts)),
        "multi_branch_ball_ratio": float(np.mean(counts > 1.0)),
        "branch_count_p50": float(np.percentile(counts, 50)),
        "branch_count_p90": float(np.percentile(counts, 90)),
        "ball_size_p50": float(np.percentile(np.asarray(ball_sizes, dtype=float), 50)),
        "ball_size_p95": float(np.percentile(np.asarray(ball_sizes, dtype=float), 95)),
    }


def run_diagnostics(dataset: pd.DataFrame, meta: pd.DataFrame, out_dir: Path, *, k_neighbors: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    local_mod = _load_analysis_script("eval_local_continuity")
    branch_mod = _load_analysis_script("eval_branch_aware_continuity")
    ds_path = out_dir / "diagnostic_dataset.parquet"
    meta_path = out_dir / "diagnostic_meta.parquet"
    dataset.to_parquet(ds_path, index=False, compression="zstd")
    meta.to_parquet(meta_path, index=False, compression="zstd")
    local_payload = local_mod.evaluate(ds_path, radii_m=[0.005, 0.01, 0.02, 0.05], k_neighbors=int(k_neighbors))
    branch_payload = branch_mod.evaluate_frames(dataset, meta, radii_m=[0.005, 0.01, 0.02], xyz_k_neighbors=int(k_neighbors))
    cluster_payload = _local_branch_metrics(dataset, meta)
    payload = {
        "local_continuity": local_payload,
        "branch_aware_continuity": branch_payload,
        "branch_clustering_beta_only": cluster_payload,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return payload


def _path_report(
    *,
    path: LayerPath,
    full_pool: pd.DataFrame,
    x_pool: pd.DataFrame,
    rho_pool: pd.DataFrame,
    jac_pool: pd.DataFrame,
    balanced: pd.DataFrame,
    diagnostics: dict[str, Any],
    x_range: tuple[float, float],
) -> dict[str, Any]:
    all10 = diagnostics.get("local_continuity", {}).get("local", {}).get("<= 10mm", {})
    branch_cluster = diagnostics.get("branch_clustering_beta_only", {})
    sigma3_stats = _percentiles(rho_pool.get("sigma3_m", pd.Series(dtype=float)).to_numpy(dtype=float), [5, 50, 95]) if len(rho_pool) else {}
    kappa_stats = _percentiles(rho_pool.get("kappa", pd.Series(dtype=float)).to_numpy(dtype=float), [50, 90, 95]) if len(rho_pool) else {}
    return {
        "path_name": str(path.name),
        "s10": float(path.s10),
        "s20": float(path.s20),
        "ds1": float(path.ds1),
        "ds2": float(path.ds2),
        "full_rows": int(len(full_pool)),
        "x_slab_rows": int(len(x_pool)),
        "rho_x_slab_rows": int(len(rho_pool)),
        "jacobian_pass_rows": int(len(jac_pool)),
        "balanced_rows": int(len(balanced)),
        "jacobian_pass_ratio_on_rho_x": float(len(jac_pool) / max(len(rho_pool), 1)),
        "x_range_m": [float(x_range[0]), float(x_range[1])],
        "workspace_full": workspace_metrics(full_pool[XYZ_COLS].to_numpy(dtype=float)) if len(full_pool) else {},
        "workspace_x_slab": workspace_metrics(x_pool[XYZ_COLS].to_numpy(dtype=float), x_range=x_range) if len(x_pool) else {},
        "workspace_balanced": workspace_metrics(balanced[XYZ_COLS].to_numpy(dtype=float), x_range=x_range) if len(balanced) else {},
        "sigma3_m": sigma3_stats,
        "kappa": kappa_stats,
        "all10_pairs": int(all10.get("pairs", 0)),
        "all10_theta_rms_p95_deg": float(all10.get("theta_rms_deg_p95", float("inf"))),
        "multi_branch_ball_ratio": float(branch_cluster.get("multi_branch_ball_ratio", 0.0)),
    }


def generate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(Path(args.config))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    x_range = (float(args.x_min), float(args.x_max))
    a_scale_rad = max(abs(float(args.a_deg[0])), abs(float(args.a_deg[1])), 1.0e-12) * math.pi / 180.0
    b_scale_rad = max(abs(float(args.b_deg[0])), abs(float(args.b_deg[1])), 1.0e-12) * math.pi / 180.0
    u_grid = build_u_grid(a_deg=tuple(args.a_deg), b_deg=tuple(args.b_deg), eta_count=int(args.eta_count))
    paths = layer_paths()
    if args.paths:
        wanted = [p.strip() for p in str(args.paths).split(",") if p.strip()]
        paths = {name: paths[name] for name in wanted}

    path_reports: list[dict[str, Any]] = []
    for path_name, path in paths.items():
        path_out = out_dir / path_name
        path_report_dir = report_dir / path_name
        path_out.mkdir(parents=True, exist_ok=True)
        path_report_dir.mkdir(parents=True, exist_ok=True)
        if bool(args.progress):
            print(json.dumps({"stage": "build_pool", "path": path_name}, ensure_ascii=False), flush=True)
        pool = build_layer_field_pool(u_grid, path)
        full_pool = attach_fk(pool, lengths_m=inputs.lengths_m, p_end_local_m=inputs.p_end_local_m, theta_sign=theta_sign)
        _write_parquet(full_pool, path_out / "full_pool.parquet")

        x_pool = full_pool[(full_pool["x_m"] >= x_range[0]) & (full_pool["x_m"] <= x_range[1])].copy().reset_index(drop=True)
        if bool(args.progress):
            print(json.dumps({"stage": "jacobian", "path": path_name, "x_rows": int(len(x_pool))}, ensure_ascii=False), flush=True)
        x_pool = attach_jacobian_gate(
            x_pool,
            path=path,
            lengths_m=inputs.lengths_m,
            p_end_local_m=inputs.p_end_local_m,
            theta_sign=theta_sign,
            a_scale_rad=a_scale_rad,
            b_scale_rad=b_scale_rad,
            delta=float(args.jacobian_delta),
            sigma3_min_m=float(args.sigma3_min_m),
            kappa_max=float(args.kappa_max),
        )
        _write_parquet(x_pool, path_out / "x1p0_1p2_pool.parquet")
        rho_pool = x_pool[x_pool["rho_deg"] >= float(args.rho_min_deg)].copy().reset_index(drop=True)
        jac_pool = rho_pool[rho_pool["jacobian_gate_pass"].astype(bool)].copy().reset_index(drop=True)
        _write_parquet(jac_pool, path_out / "jacobian_pass_pool.parquet")

        if len(jac_pool):
            balanced = select_workspace_balanced_subset(
                jac_pool[["sample_id", *XYZ_COLS, *BETA_COLS, *THETA_COLS, "path_name", "source_component", "policy_version", "u_a_deg", "u_b_deg", "u_eta", "u_a_rad", "u_b_rad", "rho_deg", "s1", "s2", "sigma1_m", "sigma2_m", "sigma3_m", "kappa", "jacobian_gate_pass"]].copy(),
                max_rows=min(int(args.balanced_rows), len(jac_pool)),
                x_range=x_range,
                x_bin_mm=float(args.x_bin_mm),
                voxel_mm=float(args.voxel_mm),
                max_per_voxel=int(args.max_per_voxel),
                seed=int(args.seed),
            )
        else:
            balanced = jac_pool.copy()
        dataset, meta = _dataset_and_meta(balanced)
        _write_parquet(balanced, path_out / f"balanced_{len(balanced)}.parquet")
        _write_parquet(dataset, path_out / "dataset.parquet")
        _write_parquet(meta, path_out / "meta.parquet")
        diagnostics = run_diagnostics(dataset, meta, path_report_dir / "diagnostics", k_neighbors=int(args.k_neighbors)) if len(dataset) else {}

        viz_dir = path_report_dir / "visualizations"
        _plot_workspace(full_pool[XYZ_COLS].to_numpy(dtype=float), viz_dir / "full_workspace_3d_multi_view.png", title=f"{path_name} full workspace")
        if len(jac_pool):
            _plot_workspace(jac_pool[XYZ_COLS].to_numpy(dtype=float), viz_dir / "jacobian_pass_workspace_3d_multi_view.png", title=f"{path_name} Jacobian pass workspace")
            _plot_axis_hist(jac_pool, viz_dir / "jacobian_pass_axis_hist.png", title=f"{path_name} Jacobian pass axis histograms")
        report = _path_report(
            path=path,
            full_pool=full_pool,
            x_pool=x_pool,
            rho_pool=rho_pool,
            jac_pool=jac_pool,
            balanced=balanced,
            diagnostics=diagnostics,
            x_range=x_range,
        )
        (path_report_dir / "path_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        path_reports.append(report)

    summary = pd.DataFrame(path_reports)
    summary_path = report_dir / "path_summary.csv"
    summary.to_csv(summary_path, index=False)
    report_payload = {
        "mode": "canonical_layer_field_u3",
        "policy_version": POLICY_VERSION,
        "config": str(args.config),
        "out_dir": str(out_dir),
        "report_dir": str(report_dir),
        "u_grid_rows_per_path": int(len(u_grid)),
        "a_deg": list(args.a_deg),
        "b_deg": list(args.b_deg),
        "eta_count": int(args.eta_count),
        "rho_min_deg": float(args.rho_min_deg),
        "sigma3_min_m": float(args.sigma3_min_m),
        "kappa_max": float(args.kappa_max),
        "path_reports": path_reports,
    }
    (report_dir / "layer_field_generation_report.json").write_text(json.dumps(report_payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    write_readme(report_dir, report_payload)
    return report_payload


def write_readme(report_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Canonical Layer-Field U3 Pilot",
        "",
        "This FK-only pilot tests a 3D canonical configuration manifold `u=(a,b,eta)` before tension labeling.",
        "",
        "## Gates",
        "",
        f"- rho min: `{report['rho_min_deg']:.3f} deg`",
        f"- sigma3 min: `{report['sigma3_min_m']:.6f} m`",
        f"- kappa max: `{report['kappa_max']:.3f}`",
        "",
        "## Path Summary",
        "",
        "| path | x rows | jac pass rows | pass ratio | balanced | all10 theta p95 deg | multi-branch ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("path_reports", []):
        lines.append(
            "| {path_name} | {x_slab_rows} | {jacobian_pass_rows} | {jacobian_pass_ratio_on_rho_x:.4f} | "
            "{balanced_rows} | {all10_theta_rms_p95_deg:.3f} | {multi_branch_ball_ratio:.4f} |".format(**row)
        )
    lines.extend(["", "## Files", ""])
    lines.append(f"- summary CSV: `{report_dir / 'path_summary.csv'}`")
    lines.append(f"- full JSON report: `{report_dir / 'layer_field_generation_report.json'}`")
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_range_step(raw: str) -> tuple[float, float, float]:
    parts = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected lo,hi,step")
    lo, hi, step = parts
    if not (lo < hi and step > 0.0):
        raise argparse.ArgumentTypeError("range must satisfy lo < hi and step > 0")
    return float(lo), float(hi), float(step)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate and diagnose a canonical 3D layer-field FK-only dataset.")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "canonical_layer_field_u3_pilot_v1")
    ap.add_argument("--report-dir", type=Path, default=REPO_ROOT / "runs" / "diagnostics" / "canonical_layer_field_u3_pilot_v1")
    ap.add_argument("--paths", default="", help="Comma-separated subset of path names; default runs all paths.")
    ap.add_argument("--a-deg", type=_parse_range_step, default=(-15.0, 15.0, 0.5))
    ap.add_argument("--b-deg", type=_parse_range_step, default=(-15.0, 15.0, 0.5))
    ap.add_argument("--eta-count", type=int, default=21)
    ap.add_argument("--x-min", type=float, default=1.0)
    ap.add_argument("--x-max", type=float, default=1.2)
    ap.add_argument("--rho-min-deg", type=float, default=2.0)
    ap.add_argument("--sigma3-min-m", type=float, default=0.002)
    ap.add_argument("--kappa-max", type=float, default=50.0)
    ap.add_argument("--jacobian-delta", type=float, default=1.0e-3)
    ap.add_argument("--balanced-rows", type=int, default=20000)
    ap.add_argument("--x-bin-mm", type=float, default=5.0)
    ap.add_argument("--voxel-mm", type=float, default=10.0)
    ap.add_argument("--max-per-voxel", type=int, default=8)
    ap.add_argument("--k-neighbors", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--progress", action="store_true")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    report = generate_dataset(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
