#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

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
EFFECTIVE_BETA_COLS = [f"effective_beta_{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
TENSION_COLS = [f"tension_{i}_n" for i in range(1, 13)]
XYZ_COLS = ["x_m", "y_m", "z_m"]
TARGET_COLS = ["target_x_m", "target_y_m", "target_z_m"]
POLICY_VERSION = "priority_grid_third_joint_first_v1"


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


def _parse_float_tuple(raw: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(raw, str):
        return tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(float(v) for v in raw)


def _linspace_deg(bounds_deg: tuple[float, float], levels: int) -> np.ndarray:
    lo, hi = float(bounds_deg[0]), float(bounds_deg[1])
    if int(levels) < 2:
        raise ValueError("grid levels must be >= 2")
    if not lo < hi:
        raise ValueError("grid degree bounds must satisfy lo < hi")
    return np.deg2rad(np.linspace(lo, hi, int(levels), dtype=float))


def build_priority_grid_pool(
    *,
    beta5_levels: int = 41,
    beta6_levels: int = 41,
    beta5_deg: tuple[float, float] = (-10.0, 10.0),
    beta6_deg: tuple[float, float] = (-10.0, 10.0),
    s2_values: Iterable[float] = (0.0, 0.25, 0.5),
    s1_values: Iterable[float] = (0.0, 0.125, 0.25),
    strict_s1_le_s2: bool = True,
) -> pd.DataFrame:
    beta5_values = _linspace_deg(tuple(beta5_deg), int(beta5_levels))
    beta6_values = _linspace_deg(tuple(beta6_deg), int(beta6_levels))
    s2 = _parse_float_tuple(tuple(s2_values))
    s1 = _parse_float_tuple(tuple(s1_values))
    rows: list[dict[str, Any]] = []
    ratio_pair_id = 0
    priority_grid_id = 0
    for s2_val in s2:
        for s1_val in s1:
            s1_val = float(s1_val)
            s2_val = float(s2_val)
            if bool(strict_s1_le_s2) and s1_val > s2_val + 1.0e-12:
                continue
            for i5, beta5 in enumerate(beta5_values):
                for i6, beta6 in enumerate(beta6_values):
                    beta = np.asarray(
                        [
                            s1_val * beta5,
                            s1_val * beta6,
                            s2_val * beta5,
                            s2_val * beta6,
                            beta5,
                            beta6,
                        ],
                        dtype=float,
                    )
                    row: dict[str, Any] = {
                        "priority_grid_id": int(priority_grid_id),
                        "priority_ratio_pair_id": int(ratio_pair_id),
                        "s1": float(s1_val),
                        "s2": float(s2_val),
                        "beta5_grid_index": int(i5),
                        "beta6_grid_index": int(i6),
                        "beta5_grid_deg": float(np.rad2deg(beta5)),
                        "beta6_grid_deg": float(np.rad2deg(beta6)),
                        "source_component": "priority_grid",
                        "priority_policy_version": POLICY_VERSION,
                    }
                    for j, value in enumerate(beta):
                        row[f"beta{j + 1}_rad"] = float(value)
                    rows.append(row)
                    priority_grid_id += 1
            ratio_pair_id += 1
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
    xyz = workspace_xyz_from_beta_rows(out[BETA_COLS].to_numpy(dtype=float), inputs, theta_sign)
    out["target_x_m"] = xyz[:, 0]
    out["target_y_m"] = xyz[:, 1]
    out["target_z_m"] = xyz[:, 2]
    out["radius_m"] = np.linalg.norm(xyz, axis=1)
    return out


def _reference_xyz(reference: pd.DataFrame) -> np.ndarray:
    if set(XYZ_COLS).issubset(reference.columns):
        return reference[XYZ_COLS].to_numpy(dtype=float)
    if set(TARGET_COLS).issubset(reference.columns):
        return reference[TARGET_COLS].to_numpy(dtype=float)
    raise ValueError("reference must contain x_m/y_m/z_m or target_x_m/target_y_m/target_z_m")


def _reference_beta(reference: pd.DataFrame) -> np.ndarray:
    if set(EFFECTIVE_BETA_COLS).issubset(reference.columns):
        return reference[EFFECTIVE_BETA_COLS].to_numpy(dtype=float)
    if set(BETA_COLS).issubset(reference.columns):
        return reference[BETA_COLS].to_numpy(dtype=float)
    raise ValueError("reference must contain effective_beta_*_rad or beta*_rad")


def _candidate_beta(candidates: pd.DataFrame) -> np.ndarray:
    if set(EFFECTIVE_BETA_COLS).issubset(candidates.columns):
        return candidates[EFFECTIVE_BETA_COLS].to_numpy(dtype=float)
    if set(BETA_COLS).issubset(candidates.columns):
        return candidates[BETA_COLS].to_numpy(dtype=float)
    raise ValueError("candidates must contain effective_beta_*_rad or beta*_rad")


def _workspace_neighbor_indices(target_xyz: np.ndarray, reference: pd.DataFrame, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    xyz = _reference_xyz(reference)
    target = np.asarray(target_xyz, dtype=float).reshape(1, 3)
    nn = NearestNeighbors(radius=float(radius_m), algorithm="auto").fit(xyz)
    dist, idx = nn.radius_neighbors(target, return_distance=True)
    order = np.argsort(dist[0]) if len(dist[0]) else np.asarray([], dtype=int)
    return np.asarray(idx[0], dtype=int)[order], np.asarray(dist[0], dtype=float)[order]


def target_inside_priority_workspace(target_xyz: np.ndarray, reference: pd.DataFrame, *, radius_m: float = 0.010) -> bool:
    idx, _dist = _workspace_neighbor_indices(target_xyz, reference, radius_m)
    return bool(idx.size > 0)


def _group_dist(candidate: np.ndarray, reference_beta: np.ndarray, cols: list[int]) -> float:
    diff = candidate.reshape(1, 6)[:, cols] - reference_beta[:, cols]
    return float(np.min(np.sqrt(np.mean(np.square(diff), axis=1))))


def select_priority_reference_candidate(
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    target_xyz: np.ndarray,
    workspace_radius_m: float = 0.010,
) -> pd.Series:
    if candidates.empty:
        raise ValueError("candidates must not be empty")
    idx, dist = _workspace_neighbor_indices(target_xyz, reference, workspace_radius_m)
    if idx.size == 0:
        raise ValueError("target is outside priority-grid workspace")
    ref_beta = _reference_beta(reference.iloc[idx])
    cand_beta = _candidate_beta(candidates)

    scored: list[tuple[tuple[float, ...], int, dict[str, float]]] = []
    for local_idx, beta in enumerate(cand_beta):
        j1 = _group_dist(beta, ref_beta, [0, 1])
        j2 = _group_dist(beta, ref_beta, [2, 3])
        j3 = _group_dist(beta, ref_beta, [4, 5])
        row = candidates.iloc[local_idx]
        key = (
            j1,
            j2,
            j3,
            float(row.get("xyz_err_m", row.get("target_xyz_err_m", 0.0))),
            float(row.get("rms_rnorm", 0.0)),
            float(row.get("max_tension", 0.0)),
        )
        scored.append(
            (
                key,
                int(local_idx),
                {
                    "priority_reference_neighbor_count": float(idx.size),
                    "priority_reference_nearest_xyz_dist_m": float(dist[0]),
                    "priority_reference_joint1_dist": j1,
                    "priority_reference_joint2_dist": j2,
                    "priority_reference_joint3_dist": j3,
                },
            )
        )
    _key, best_idx, meta = sorted(scored, key=lambda item: (item[0], item[1]))[0]
    out = candidates.iloc[best_idx].copy()
    for key, value in meta.items():
        out[key] = int(value) if key == "priority_reference_neighbor_count" else float(value)
    return out


def priority_grid_to_rows(
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
    pool_row: pd.Series,
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
        "source_component": "priority_grid",
        "priority_policy_version": POLICY_VERSION,
        "priority_grid_id": int(pool_row["priority_grid_id"]),
        "priority_ratio_pair_id": int(pool_row["priority_ratio_pair_id"]),
        "s1": float(pool_row["s1"]),
        "s2": float(pool_row["s2"]),
        "beta5_grid_index": int(pool_row["beta5_grid_index"]),
        "beta6_grid_index": int(pool_row["beta6_grid_index"]),
        "beta5_grid_deg": float(pool_row["beta5_grid_deg"]),
        "beta6_grid_deg": float(pool_row["beta6_grid_deg"]),
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
    flags = np.zeros(12, dtype=int) if case_flag_12 is None else np.asarray(case_flag_12, dtype=int).reshape(-1)
    for j in range(12):
        meta[f"case_{j + 1}"] = int(flags[j]) if j < flags.size else 0
    return row, meta


def _label_beta(
    *,
    sample_id: int,
    beta: np.ndarray,
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
    row, meta = priority_grid_to_rows(
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
        pool_row=pool_row,
        case_flag_12=cache.case_flag_12,
    )
    ok = bool(label.ok and np.isfinite(label.T_base_12).all() and float(meta["rms_rnorm"]) <= float(rms_thresh))
    return ok, row, meta


def workspace_coverage(xyz: np.ndarray) -> dict[str, Any]:
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    r = np.linalg.norm(xyz, axis=1)
    yz = xyz[:, 1:3]
    area = float("nan")
    try:
        from scipy.spatial import ConvexHull

        if len(yz) >= 3:
            area = float(ConvexHull(yz).volume)
    except Exception:
        area = float("nan")
    return {
        "n": int(len(xyz)),
        "x_minmax_mm": [float(xyz[:, 0].min() * 1000.0), float(xyz[:, 0].max() * 1000.0)],
        "y_minmax_mm": [float(xyz[:, 1].min() * 1000.0), float(xyz[:, 1].max() * 1000.0)],
        "z_minmax_mm": [float(xyz[:, 2].min() * 1000.0), float(xyz[:, 2].max() * 1000.0)],
        "radius_minmax_mm": [float(r.min() * 1000.0), float(r.max() * 1000.0)],
        "yz_hull_area_mm2": None if not np.isfinite(area) else float(area * 1.0e6),
        "yz_span_product_mm2": float((xyz[:, 1].max() - xyz[:, 1].min()) * (xyz[:, 2].max() - xyz[:, 2].min()) * 1.0e6),
    }


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
    all10 = continuity.get("groups", {}).get("all_xyz_<=10mm", {})
    beta_close = continuity.get("groups", {}).get("xyz_<=10mm_beta_close", {})
    hard_gate = bool(
        np.isfinite(tension).all()
        and (tension >= 0.0).all()
        and (tension <= 2000.0).all()
        and float(np.percentile(rms, 95)) <= 0.06
        and float(np.percentile(xyz_err, 95)) <= 0.004
    )
    gate = {
        "hard_gate_passed": hard_gate,
        "workspace_yz_hull_area_mm2": workspace_coverage(dataset[XYZ_COLS].to_numpy(dtype=float)).get("yz_hull_area_mm2"),
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
        "workspace_coverage": workspace_coverage(dataset[XYZ_COLS].to_numpy(dtype=float)),
        "continuity": continuity,
        "branch_clustering": {k: v for k, v in clustering.items() if k != "per_ball"},
        "oracle_floor": oracle,
    }
    (out_dir / "priority_grid_diagnostics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return payload


def generate_priority_grid_dataset(
    *,
    cfg: dict[str, Any],
    out_dir: Path,
    beta5_levels: int = 41,
    beta6_levels: int = 41,
    beta5_deg: tuple[float, float] = (-10.0, 10.0),
    beta6_deg: tuple[float, float] = (-10.0, 10.0),
    s2_values: Iterable[float] = (0.0, 0.25, 0.5),
    s1_values: Iterable[float] = (0.0, 0.125, 0.25),
    strict_s1_le_s2: bool = True,
    limit_rows: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_robot_inputs(cfg)
    model = QuasiStaticModel(cfg, inputs)
    seed = int(cfg.get("sampling", {}).get("rng_seed", 20260610))
    t0 = time.time()
    pool = build_priority_grid_pool(
        beta5_levels=int(beta5_levels),
        beta6_levels=int(beta6_levels),
        beta5_deg=tuple(beta5_deg),
        beta6_deg=tuple(beta6_deg),
        s2_values=s2_values,
        s1_values=s1_values,
        strict_s1_le_s2=bool(strict_s1_le_s2),
    )
    if limit_rows is not None:
        pool = pool.iloc[: int(limit_rows)].copy().reset_index(drop=True)
    pool = attach_workspace_targets(pool, inputs, model.theta_sign)
    (out_dir / "priority_grid_pool.parquet").parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out_dir / "priority_grid_pool.parquet", index=False)

    pso_cfg = cfg["pso"]
    rms_thresh = float(cfg.get("dataset", {}).get("rms_rnorm_threshold", 0.06))
    rows: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    failures = 0
    for sample_id, pool_row in enumerate(tqdm(pool.itertuples(index=False), total=len(pool), desc="priority_grid")):
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
        "mode": "priority_grid",
        "policy_version": POLICY_VERSION,
        "requested_rows": int(len(pool)),
        "selected_rows": int(len(dataset)),
        "label_failures": int(failures),
        "elapsed_s": float(time.time() - t0),
        "sec_per_selected_row": float((time.time() - t0) / max(len(dataset), 1)),
        "grid": {
            "beta5_levels": int(beta5_levels),
            "beta6_levels": int(beta6_levels),
            "beta5_deg": [float(beta5_deg[0]), float(beta5_deg[1])],
            "beta6_deg": [float(beta6_deg[0]), float(beta6_deg[1])],
            "s1_values": [float(v) for v in s1_values],
            "s2_values": [float(v) for v in s2_values],
            "strict_s1_le_s2": bool(strict_s1_le_s2),
            "ratio_pair_count": int(pool["priority_ratio_pair_id"].nunique()) if "priority_ratio_pair_id" in pool.columns else 0,
        },
        "workspace_coverage": workspace_coverage(pool[TARGET_COLS].to_numpy(dtype=float)),
        "gate": diagnostics.get("gate", {}),
    }
    (out_dir / "priority_grid_generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("data/priority_grid_third_joint_first_v1"))
    ap.add_argument("--beta5-levels", type=int, default=41)
    ap.add_argument("--beta6-levels", type=int, default=41)
    ap.add_argument("--beta5-deg", default="-10,10")
    ap.add_argument("--beta6-deg", default="-10,10")
    ap.add_argument("--s2-values", default="0,0.25,0.5")
    ap.add_argument("--s1-values", default="0,0.125,0.25")
    ap.add_argument("--allow-s1-gt-s2", action="store_true")
    ap.add_argument("--limit-rows", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    report = generate_priority_grid_dataset(
        cfg=cfg,
        out_dir=args.out_dir,
        beta5_levels=int(args.beta5_levels),
        beta6_levels=int(args.beta6_levels),
        beta5_deg=tuple(_parse_float_tuple(args.beta5_deg)),
        beta6_deg=tuple(_parse_float_tuple(args.beta6_deg)),
        s2_values=_parse_float_tuple(args.s2_values),
        s1_values=_parse_float_tuple(args.s1_values),
        strict_s1_le_s2=not bool(args.allow_s1_gt_s2),
        limit_rows=args.limit_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
