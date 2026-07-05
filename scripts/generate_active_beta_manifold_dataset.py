#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from quasi_exp.model.kinematics import forward_kinematics  # noqa: E402
from quasi_exp.model.quasi_static import QuasiStaticModel  # noqa: E402
from quasi_exp.model.sampling import beta_to_theta, effective_beta_from_theta  # noqa: E402
from quasi_exp.opt.tension_labeler import solve_tension_label  # noqa: E402


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
RAW_BETA_COLS = [f"raw_beta{i}_rad" for i in range(1, 7)]
EFFECTIVE_BETA_COLS = [f"effective_beta_{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
TENSION_COLS = [f"tension_{i}_n" for i in range(1, 13)]
TARGET_COLS = ["target_x_m", "target_y_m", "target_z_m"]
POLICY_VERSION = "active_beta_manifold_distal_v1"
VARIANTS = ("manifold_3d_full_angle", "filtered_6d_distal_cloud", "manifold_3d_quadrant")
VARIANT_SEED_OFFSETS = {
    "manifold_3d_full_angle": 1103,
    "filtered_6d_distal_cloud": 2207,
    "manifold_3d_quadrant": 3301,
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def beta_bounds_from_ranges(beta_ranges_rad: dict[str, Any]) -> np.ndarray:
    bounds = np.zeros((6, 2), dtype=float)
    for i in range(6):
        key = f"beta{i + 1}"
        if key not in beta_ranges_rad:
            raise ValueError(f"sampling.beta_ranges_rad missing {key}")
        lo, hi = beta_ranges_rad[key]
        bounds[i, 0] = float(lo)
        bounds[i, 1] = float(hi)
        if not bounds[i, 0] < bounds[i, 1]:
            raise ValueError(f"{key} range must satisfy lo < hi")
    return bounds


def beta_scale_from_bounds(bounds: np.ndarray) -> np.ndarray:
    b = np.asarray(bounds, dtype=float).reshape(6, 2)
    return np.maximum(np.maximum(np.abs(b[:, 0]), np.abs(b[:, 1])), 1.0e-12)


def _sobol_unit(n: int, d: int, seed: int) -> np.ndarray:
    if int(n) <= 0:
        return np.zeros((0, int(d)), dtype=float)
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=int(d), scramble=True, seed=int(seed))
        m = int(np.ceil(np.log2(max(1, int(n)))))
        return np.asarray(sampler.random_base2(m), dtype=float)[: int(n)]
    except Exception:
        rng = np.random.default_rng(int(seed))
        return rng.random((int(n), int(d)))


def _scale_unit(unit: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    u = np.asarray(unit, dtype=float)
    return lo.reshape(1, -1) + np.clip(u, 0.0, 1.0) * (hi - lo).reshape(1, -1)


def build_beta_manifold_pool(
    *,
    variant: str,
    pool_size: int,
    beta_bounds: np.ndarray,
    seed: int,
    proximal_abs_ratio_max: float = 0.45,
    distal_abs_ratio_min: float = 0.55,
) -> pd.DataFrame:
    variant = str(variant)
    if variant not in VARIANTS:
        raise ValueError(f"unsupported beta manifold variant: {variant}")
    n = int(pool_size)
    if n <= 0:
        raise ValueError("pool_size must be > 0")
    bounds = np.asarray(beta_bounds, dtype=float).reshape(6, 2)
    scale = beta_scale_from_bounds(bounds)

    rows: list[dict[str, Any]] = []
    if variant in {"manifold_3d_full_angle", "manifold_3d_quadrant"}:
        u = _sobol_unit(n, 3, int(seed))
        rho = 0.25 + 0.75 * u[:, 0]
        if variant == "manifold_3d_quadrant":
            phi = (np.pi / 2.0) * u[:, 1]
        else:
            phi = -np.pi + 2.0 * np.pi * u[:, 1]
        q = 0.5 * u[:, 2]
        distal = np.stack([np.cos(phi), np.sin(phi)], axis=1) * (rho * scale[4]).reshape(-1, 1)
        beta = np.zeros((n, 6), dtype=float)
        beta[:, 4:6] = distal
        beta[:, 2:4] = q.reshape(-1, 1) * distal
        beta[:, 0:2] = 0.5 * q.reshape(-1, 1) * distal
        beta = np.clip(beta, bounds[:, 0][None, :], bounds[:, 1][None, :])
        for idx in range(n):
            row = {
                "source_pool_index": int(idx),
                "source_component": variant,
                "manifold_variant": variant,
                "latent_rho": float(rho[idx]),
                "latent_phi": float(phi[idx]),
                "latent_q": float(q[idx]),
            }
            for j in range(6):
                row[f"beta{j + 1}_rad"] = float(beta[idx, j])
            rows.append(row)
    else:
        rng = np.random.default_rng(int(seed))
        prox_max = float(np.clip(proximal_abs_ratio_max, 0.0, 1.0))
        distal_min = float(np.clip(distal_abs_ratio_min, 0.0, 1.0))
        beta = np.zeros((n, 6), dtype=float)
        prox_mag = rng.uniform(0.0, prox_max, size=(n, 4))
        distal_mag = rng.uniform(distal_min, 1.0, size=(n, 2))
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=(n, 6))
        beta[:, :4] = signs[:, :4] * prox_mag * scale[:4][None, :]
        beta[:, 4:] = signs[:, 4:] * distal_mag * scale[4:][None, :]
        beta = np.clip(beta, bounds[:, 0][None, :], bounds[:, 1][None, :])
        for idx in range(n):
            row = {
                "source_pool_index": int(idx),
                "source_component": variant,
                "manifold_variant": variant,
                "latent_rho": float(np.linalg.norm(beta[idx, 4:6] / scale[4:6]) / np.sqrt(2.0)),
                "latent_phi": float(np.arctan2(beta[idx, 5], beta[idx, 4])),
                "latent_q": float(np.linalg.norm(beta[idx, 2:4] / scale[2:4]) / max(np.linalg.norm(beta[idx, 4:6] / scale[4:6]), 1.0e-12)),
            }
            for j in range(6):
                row[f"beta{j + 1}_rad"] = float(beta[idx, j])
            rows.append(row)
    return pd.DataFrame(rows)


def workspace_xyz_from_beta_rows(beta_rows: np.ndarray, inputs: Any, theta_sign: float) -> np.ndarray:
    beta = np.asarray(beta_rows, dtype=float).reshape(-1, 6)
    xyz = np.zeros((len(beta), 3), dtype=float)
    for i, row in enumerate(beta):
        theta_raw = beta_to_theta(row)
        p_xyz, _ = forward_kinematics(theta_raw, inputs.lengths_m, inputs.p_end_local_m, theta_sign=float(theta_sign))
        xyz[i] = np.asarray(p_xyz, dtype=float).reshape(3)
    return xyz


def attach_workspace_targets(pool: pd.DataFrame, inputs: Any, theta_sign: float) -> pd.DataFrame:
    out = pool.copy().reset_index(drop=True)
    beta = out[BETA_COLS].to_numpy(dtype=float)
    xyz = workspace_xyz_from_beta_rows(beta, inputs, theta_sign)
    out["target_x_m"] = xyz[:, 0]
    out["target_y_m"] = xyz[:, 1]
    out["target_z_m"] = xyz[:, 2]
    out["radius_m"] = np.linalg.norm(xyz, axis=1)
    return out


def apply_local_injectivity_filter(
    pool: pd.DataFrame,
    *,
    beta_scale_rad: np.ndarray,
    radius_m: float = 0.01,
    beta_far_threshold_norm: float = 0.15,
    min_neighbors: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if pool.empty:
        return pool.copy(), {"input_rows": 0, "kept_rows": 0, "rejected_beta_far_count": 0}
    xyz = pool[TARGET_COLS].to_numpy(dtype=float)
    beta = pool[BETA_COLS].to_numpy(dtype=float)
    scale = np.asarray(beta_scale_rad, dtype=float).reshape(6)
    nn = NearestNeighbors(radius=float(radius_m), algorithm="auto").fit(xyz)
    neigh = nn.radius_neighbors(xyz, return_distance=False)
    keep = np.ones(len(pool), dtype=bool)
    counts = np.zeros(len(pool), dtype=int)
    max_d = np.zeros(len(pool), dtype=float)
    for i, raw in enumerate(neigh):
        idx = np.asarray([int(v) for v in raw.tolist() if int(v) != i], dtype=int)
        counts[i] = int(idx.size)
        if idx.size == 0:
            continue
        d = np.sqrt(np.mean(np.square((beta[idx] - beta[i]) / scale.reshape(1, 6)), axis=1))
        max_d[i] = float(np.max(d))
        if idx.size >= int(min_neighbors) and max_d[i] > float(beta_far_threshold_norm):
            keep[i] = False
    out = pool.loc[keep].copy().reset_index(drop=True)
    out["injectivity_neighbor_count"] = counts[keep]
    out["injectivity_max_beta_dist_norm"] = max_d[keep]
    out["injectivity_radius_m"] = float(radius_m)
    out["injectivity_beta_far_threshold_norm"] = float(beta_far_threshold_norm)
    report = {
        "input_rows": int(len(pool)),
        "kept_rows": int(len(out)),
        "rejected_beta_far_count": int(np.sum(~keep)),
        "keep_ratio": float(len(out) / max(len(pool), 1)),
        "radius_m": float(radius_m),
        "beta_far_threshold_norm": float(beta_far_threshold_norm),
        "min_neighbors": int(min_neighbors),
        "neighbor_count_p50": float(np.percentile(counts, 50)) if len(counts) else 0.0,
        "neighbor_count_p95": float(np.percentile(counts, 95)) if len(counts) else 0.0,
        "max_beta_dist_norm_p95": float(np.percentile(max_d, 95)) if len(max_d) else 0.0,
    }
    return out, report


def _workspace_bins(xyz: np.ndarray, *, radius_bins: int, z_bins: int, angle_bins: int) -> list[tuple[int, int, int]]:
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)

    def qbin(values: np.ndarray, n_bins: int) -> np.ndarray:
        n_bins = max(1, int(n_bins))
        if values.size == 0 or n_bins == 1:
            return np.zeros(values.size, dtype=int)
        qs = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        return np.searchsorted(qs, values, side="right").astype(int)

    rb = qbin(np.linalg.norm(xyz, axis=1), radius_bins)
    zb = qbin(xyz[:, 2], z_bins)
    angle = np.arctan2(xyz[:, 2], xyz[:, 1])
    ab = np.floor(((angle + np.pi) / (2.0 * np.pi)) * max(1, int(angle_bins))).astype(int)
    ab = np.clip(ab, 0, max(1, int(angle_bins)) - 1)
    return [(int(a), int(b), int(c)) for a, b, c in zip(rb.tolist(), zb.tolist(), ab.tolist())]


def select_workspace_balanced(pool: pd.DataFrame, *, num_samples: int, radius_bins: int, z_bins: int, angle_bins: int, seed: int) -> pd.DataFrame:
    if len(pool) < int(num_samples):
        raise RuntimeError(f"only {len(pool)} locally-injective rows available for {num_samples} samples")
    work = pool.copy().reset_index(drop=True)
    xyz = work[TARGET_COLS].to_numpy(dtype=float)
    bins = _workspace_bins(xyz, radius_bins=radius_bins, z_bins=z_bins, angle_bins=angle_bins)
    work["_workspace_balance_bin"] = [f"{a},{b},{c}" for a, b, c in bins]
    work["_rand"] = np.random.default_rng(int(seed)).random(len(work))
    ordered = work.sort_values(["_workspace_balance_bin", "_rand"], kind="mergesort")
    buckets = {key: block.index.to_list() for key, block in ordered.groupby("_workspace_balance_bin", sort=True)}
    keys = sorted(buckets)
    selected: list[int] = []
    cursor = 0
    while len(selected) < int(num_samples) and keys:
        key = keys[cursor % len(keys)]
        vals = buckets[key]
        if vals:
            selected.append(int(vals.pop(0)))
        if not vals:
            keys.remove(key)
            if not keys:
                break
            cursor = cursor % len(keys)
        else:
            cursor += 1
    out = work.loc[selected[: int(num_samples)]].copy().reset_index(drop=True)
    out = out.drop(columns=[c for c in ["_workspace_balance_bin", "_rand"] if c in out.columns])
    out["active_target_id"] = np.arange(len(out), dtype=int)
    return out


def beta_manifold_to_rows(
    *,
    sample_id: int,
    beta: np.ndarray,
    theta: np.ndarray,
    tension: np.ndarray,
    xyz: np.ndarray,
    rms_rnorm: float,
    mean_rnorm2: float,
    best_cost: float,
    iters_used: int,
    evals: int,
    pso_seed: int,
    elapsed_s: float,
    variant: str,
    latent: dict[str, Any] | None = None,
    injectivity: dict[str, Any] | None = None,
    case_flag_12: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    xyz = np.asarray(xyz, dtype=float).reshape(3)
    theta = np.asarray(theta, dtype=float).reshape(30)
    tension = np.asarray(tension, dtype=float).reshape(12)
    effective = effective_beta_from_theta(theta.reshape(1, 30))[0]
    raw_beta = np.asarray(beta, dtype=float).reshape(6)
    row: dict[str, Any] = {"sample_id": int(sample_id), "x_m": float(xyz[0]), "y_m": float(xyz[1]), "z_m": float(xyz[2])}
    for i, value in enumerate(theta):
        row[f"theta_{i + 1}_rad"] = float(value)
    for j, value in enumerate(tension):
        row[f"tension_{j + 1}_n"] = float(value)
    meta: dict[str, Any] = {
        "sample_id": int(sample_id),
        "source_component": str(variant),
        "manifold_variant": str(variant),
        "active_branch_policy_version": POLICY_VERSION,
        "target_x_m": float(xyz[0]),
        "target_y_m": float(xyz[1]),
        "target_z_m": float(xyz[2]),
        "achieved_x_m": float(xyz[0]),
        "achieved_y_m": float(xyz[1]),
        "achieved_z_m": float(xyz[2]),
        "xyz_err_m": 0.0,
        "rms_rnorm": float(rms_rnorm),
        "mean_rnorm2": float(mean_rnorm2),
        "max_tension": float(np.max(tension)),
        "best_cost": float(best_cost),
        "iters_used": int(iters_used),
        "evals": int(evals),
        "pso_seed": int(pso_seed),
        "elapsed_s": float(elapsed_s),
        "segmented_success": bool(np.isfinite(tension).all() and float(rms_rnorm) <= 0.06),
        "effective_beta_source": "theta_odd_even_mean",
        "active_branch_selected": True,
        "active_graph_component_id": 0,
        "graph_component_id": 0,
    }
    for i, value in enumerate(raw_beta):
        meta[f"raw_beta{i + 1}_rad"] = float(value)
    for i, value in enumerate(effective):
        meta[f"beta{i + 1}_rad"] = float(value)
        meta[f"effective_beta_{i + 1}_rad"] = float(value)
    for key, value in dict(latent or {}).items():
        meta[str(key)] = float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value
    for key, value in dict(injectivity or {}).items():
        meta[str(key)] = float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value
    flags = np.zeros(12, dtype=int) if case_flag_12 is None else np.asarray(case_flag_12, dtype=int).reshape(-1)
    for j in range(12):
        meta[f"case_{j + 1}"] = int(flags[j]) if j < flags.size else 0
    return row, meta


def _label_beta(
    *,
    sample_id: int,
    beta: np.ndarray,
    variant: str,
    pool_row: pd.Series,
    model: QuasiStaticModel,
    inputs: Any,
    pso_cfg: dict[str, Any],
    pso_seed: int,
    rms_thresh: float,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    t0 = time.time()
    raw_beta = np.asarray(beta, dtype=float).reshape(6)
    theta_raw = beta_to_theta(raw_beta)
    theta = theta_raw * float(model.theta_sign)
    xyz, _ = forward_kinematics(theta_raw, inputs.lengths_m, inputs.p_end_local_m, theta_sign=model.theta_sign)
    cache = model.build_cache(theta_raw)
    label = solve_tension_label(model=model, cache=cache, pso_cfg=pso_cfg, pso_seed=int(pso_seed), rms_thresh=float(rms_thresh))
    latent = {c: pool_row[c] for c in ["latent_rho", "latent_phi", "latent_q"] if c in pool_row.index}
    injectivity = {
        c: pool_row[c]
        for c in ["injectivity_neighbor_count", "injectivity_max_beta_dist_norm", "injectivity_radius_m", "injectivity_beta_far_threshold_norm"]
        if c in pool_row.index
    }
    row, meta = beta_manifold_to_rows(
        sample_id=sample_id,
        beta=raw_beta,
        theta=theta,
        tension=label.T_base_12,
        xyz=np.asarray(xyz, dtype=float).reshape(3),
        rms_rnorm=float(label.meta.get("rms_rnorm", float("inf"))),
        mean_rnorm2=float(label.meta.get("mean_rnorm2", float("inf"))),
        best_cost=float(label.meta.get("best_cost", 0.0)),
        iters_used=int(label.meta.get("iters_used", 0)),
        evals=int(label.meta.get("evals", 0)),
        pso_seed=int(pso_seed),
        elapsed_s=float(time.time() - t0),
        variant=variant,
        latent=latent,
        injectivity=injectivity,
        case_flag_12=cache.case_flag_12,
    )
    ok = bool(label.ok and np.isfinite(label.T_base_12).all() and float(meta["rms_rnorm"]) <= float(rms_thresh))
    return ok, row, meta


def diagnose_and_gate(dataset: pd.DataFrame, meta: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    continuity_mod = _load_script("eval_branch_aware_continuity")
    clustering_mod = _load_script("eval_branch_clustering")
    oracle_mod = _load_script("eval_oracle_floor")
    continuity = continuity_mod.evaluate_frames(dataset, meta)
    clustering = clustering_mod.evaluate_frames(dataset, meta)
    oracle = oracle_mod.evaluate_frames(dataset, meta, branch_column="active_graph_component_id")
    tension = dataset[TENSION_COLS].to_numpy(dtype=float)
    rms = meta["rms_rnorm"].to_numpy(dtype=float) if "rms_rnorm" in meta.columns else np.full(len(meta), np.inf)
    xyz_err = meta["xyz_err_m"].to_numpy(dtype=float) if "xyz_err_m" in meta.columns else np.full(len(meta), np.inf)
    hard_gate = bool(
        np.isfinite(tension).all()
        and (tension >= 0.0).all()
        and (tension <= 2000.0).all()
        and float(np.percentile(rms, 95)) <= 0.06
        and float(np.percentile(xyz_err, 95)) <= 0.004
    )
    all10 = continuity.get("groups", {}).get("all_xyz_<=10mm", {})
    beta_close = continuity.get("groups", {}).get("xyz_<=10mm_beta_close", {})
    gate = {
        "hard_gate_passed": hard_gate,
        "xyz_err_m_p95": float(np.percentile(xyz_err, 95)) if xyz_err.size else float("inf"),
        "xyz_err_m_max": float(np.max(xyz_err)) if xyz_err.size else float("inf"),
        "rms_rnorm_q95": float(np.percentile(rms, 95)) if rms.size else float("inf"),
        "all10_theta_p95_deg": float(all10.get("theta_rms_deg_p95", float("inf"))),
        "all10_tension_p95_n": float(all10.get("tension_mae_n_p95", float("inf"))),
        "beta_close_tension_p95_n": float(beta_close.get("tension_mae_n_p95", float("inf"))),
        "multi_branch_ball_ratio": float(clustering.get("multi_branch_ball_ratio", float("inf"))),
        "xyz_nn_tension_mae_n": float(oracle.get("xyz_nn_oracle", {}).get("tension_mae_n", float("inf"))),
        "beta_nn_tension_mae_n": float(oracle.get("beta_nn_oracle", {}).get("tension_mae_n", float("inf"))),
    }
    payload = {
        "gate": gate,
        "continuity": continuity,
        "branch_clustering": {k: v for k, v in clustering.items() if k != "per_ball"},
        "oracle_floor": oracle,
    }
    (out_dir / "active_diagnostics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return payload


def generate_beta_manifold_dataset(
    *,
    cfg: dict[str, Any],
    variant: str,
    num_samples: int,
    pool_size: int,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_robot_inputs(cfg)
    model = QuasiStaticModel(cfg, inputs)
    sampling_cfg = cfg.get("sampling", {})
    manifold_cfg = cfg.get("active_beta_manifold", {})
    seed = int(sampling_cfg.get("rng_seed", manifold_cfg.get("rng_seed", 20260609)))
    bounds = beta_bounds_from_ranges(sampling_cfg["beta_ranges_rad"])
    beta_scale = beta_scale_from_bounds(bounds)
    t0 = time.time()
    raw_pool = build_beta_manifold_pool(
        variant=variant,
        pool_size=int(pool_size),
        beta_bounds=bounds,
        seed=seed + int(VARIANT_SEED_OFFSETS.get(variant, 0)),
        proximal_abs_ratio_max=float(manifold_cfg.get("proximal_abs_ratio_max", 0.45)),
        distal_abs_ratio_min=float(manifold_cfg.get("distal_abs_ratio_min", 0.55)),
    )
    pool = attach_workspace_targets(raw_pool, inputs, model.theta_sign)
    filtered, inject_report = apply_local_injectivity_filter(
        pool,
        beta_scale_rad=beta_scale,
        radius_m=float(manifold_cfg.get("injectivity_radius_m", 0.01)),
        beta_far_threshold_norm=float(manifold_cfg.get("beta_far_threshold_norm", 0.15)),
        min_neighbors=int(manifold_cfg.get("injectivity_min_neighbors", 1)),
    )
    selected = select_workspace_balanced(
        filtered,
        num_samples=int(num_samples),
        radius_bins=int(manifold_cfg.get("radius_bins", 5)),
        z_bins=int(manifold_cfg.get("z_bins", 5)),
        angle_bins=int(manifold_cfg.get("angle_bins", 8)),
        seed=seed + 17,
    )
    pso_cfg = cfg["pso"]
    rms_thresh = float(cfg.get("dataset", {}).get("rms_rnorm_threshold", 0.06))
    rows: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    failures = 0
    for sample_id, pool_row in enumerate(tqdm(selected.itertuples(index=False), total=len(selected), desc=f"beta_manifold:{variant}")):
        series = pd.Series(pool_row._asdict())
        beta = np.asarray([getattr(pool_row, f"beta{i}_rad") for i in range(1, 7)], dtype=float)
        ok, row, meta = _label_beta(
            sample_id=sample_id,
            beta=beta,
            variant=variant,
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
        rows.append(row)
        metas.append(meta)
    if len(rows) < int(num_samples):
        raise RuntimeError(f"{variant} labeled only {len(rows)}/{num_samples} rows; failures={failures}")
    dataset = pd.DataFrame(rows).reset_index(drop=True)
    meta = pd.DataFrame(metas).reset_index(drop=True)
    dataset["sample_id"] = np.arange(len(dataset), dtype=int)
    meta["sample_id"] = np.arange(len(meta), dtype=int)
    dataset.to_parquet(out_dir / "dataset.parquet", index=False)
    meta.to_parquet(out_dir / "dataset_meta.parquet", index=False)
    diagnostics = diagnose_and_gate(dataset, meta, out_dir / "diagnostics")
    report = {
        "mode": "active_beta_manifold",
        "variant": variant,
        "policy_version": POLICY_VERSION,
        "num_samples_requested": int(num_samples),
        "pool_size": int(pool_size),
        "raw_pool_rows": int(len(raw_pool)),
        "locally_injective_rows": int(len(filtered)),
        "selected_rows": int(len(dataset)),
        "label_failures": int(failures),
        "elapsed_s": float(time.time() - t0),
        "sec_per_selected_row": float((time.time() - t0) / max(len(dataset), 1)),
        "injectivity": inject_report,
        "gate": diagnostics["gate"],
    }
    (out_dir / "active_generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--pool-size", type=int, default=20000)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = args.out_dir or Path(cfg.get("dataset", {}).get("out_dir", "data/active_beta_manifold_distal_v1")) / str(args.variant)
    report = generate_beta_manifold_dataset(
        cfg=cfg,
        variant=str(args.variant),
        num_samples=int(args.num_samples),
        pool_size=int(args.pool_size),
        out_dir=out_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
