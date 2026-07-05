from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from quasi_exp.io import load_config, load_robot_inputs
from quasi_exp.model.sampling import (
    beta_to_theta,
    build_mixed_beta_tasks,
    build_standard_sweep_tasks,
    validate_generation_strategy,
)
from quasi_exp.model.kinematics import forward_kinematics
from quasi_exp.model.quasi_static import QuasiStaticModel
from quasi_exp.opt.pso_inverse import solve_inverse_joint_pso
from quasi_exp.opt.tension_labeler import solve_tension_label
from subprocess_pool import poll_results, start_workers, stop_workers, submit_task


def _flatten_sample_rows(sample: dict) -> tuple[dict, dict]:
    x, y, z = sample["x_m"], sample["y_m"], sample["z_m"]
    theta = sample["theta_rad"]
    tension = sample["tension_n"]
    meta = sample["meta"]
    beta = sample.get("beta6_rad")

    row = {"sample_id": int(sample.get("sample_id", -1)), "x_m": x, "y_m": y, "z_m": z}
    row.update({f"theta_{i+1}_rad": float(theta[i]) for i in range(30)})
    row.update({f"tension_{j+1}_n": float(tension[j]) for j in range(12)})

    mrow = {
        "sample_id": int(sample.get("sample_id", -1)),
        "seed": int(sample.get("seed", 0)),
        "rms_rnorm": float(meta["rms_rnorm"]),
        "mean_rnorm2": float(meta["mean_rnorm2"]),
        "max_tension": float(meta["max_tension"]),
        "best_cost": float(meta["best_cost"]),
        "iters_used": int(meta["iters_used"]),
        "evals": int(meta["evals"]),
        "pso_seed": int(meta["pso_seed"]),
        "elapsed_s": float(meta["elapsed_s"]),
    }
    if beta is not None:
        mrow.update({f"beta{i+1}_rad": float(beta[i]) for i in range(6)})
    for key in ["target_x_m", "target_y_m", "target_z_m", "xyz_err_m"]:
        if key in meta:
            mrow[key] = float(meta[key])
    for key in [
        "pso_elapsed_s",
        "pso_rms_rnorm",
        "pso_mean_rnorm2",
        "pso_max_tension",
        "pso_best_cost",
        "canonical_elapsed_s",
        "canonical_objective",
        "canonical_rms_rnorm",
        "canonical_mean_rnorm2",
        "canonical_max_tension",
        "segmented_elapsed_s",
        "segmented_rms_rnorm",
        "segmented_mean_rnorm2",
        "segmented_max_tension",
        "segmented_section_third_rms_rnorm",
        "segmented_section_second_rms_rnorm",
        "segmented_section_first_rms_rnorm",
        "segmented_section_third_elapsed_s",
        "segmented_section_second_elapsed_s",
        "segmented_section_first_elapsed_s",
    ]:
        if key in meta:
            mrow[key] = float(meta[key])
    for key in [
        "canonical_enabled",
        "canonical_success",
        "canonical_adopted",
        "segmented_success",
    ]:
        if key in meta:
            mrow[key] = bool(meta[key])
    for key in [
        "canonical_nit",
        "canonical_nfev",
        "canonical_source_index",
        "segmented_section_third_nfev",
        "segmented_section_second_nfev",
        "segmented_section_first_nfev",
    ]:
        if key in meta:
            mrow[key] = int(meta[key])
    for key in ["canonical_method", "tension_solver_method"]:
        if key in meta:
            mrow[key] = str(meta[key])
    for key in ["scan_axis", "scan_axis_idx", "scan_sign", "scan_level", "scan_angle_rad", "scan_angle_deg", "retry_count"]:
        if key in meta:
            value = meta[key]
            if key in {"scan_axis"}:
                mrow[key] = str(value)
            elif key in {"scan_axis_idx", "scan_sign", "scan_level", "retry_count"}:
                mrow[key] = int(value)
            else:
                mrow[key] = float(value)
    for key in ["source_component"]:
        if key in meta:
            mrow[key] = str(meta[key])
    for key in ["source_component_idx"]:
        if key in meta:
            mrow[key] = int(meta[key])
    for key in ["beta_group1_norm", "beta_group2_norm", "beta_group3_norm", "distal_preference_score", "radius_m"]:
        if key in meta:
            mrow[key] = float(meta[key])
    case_flags = meta.get("case_flag_12", [])
    for j in range(12):
        mrow[f"case_{j+1}"] = int(case_flags[j]) if j < len(case_flags) else 0

    return row, mrow


def _sample_beta(rng: np.random.Generator, beta_ranges: dict) -> np.ndarray:
    beta = np.zeros(6, dtype=float)
    for i in range(6):
        lo, hi = beta_ranges[f"beta{i+1}"]
        beta[i] = rng.uniform(float(lo), float(hi))
    return beta


def _sample_xyz_target(rng: np.random.Generator, xyz_ranges_m: dict) -> np.ndarray:
    target = np.zeros(3, dtype=float)
    for axis_idx, axis in enumerate(["x", "y", "z"]):
        key = f"{axis}_m"
        if key not in xyz_ranges_m:
            raise ValueError(f"sampling.xyz_ranges_m missing {key}")
        lo, hi = xyz_ranges_m[key]
        target[axis_idx] = rng.uniform(float(lo), float(hi))
    return target


def _workspace_xyz_from_beta_rows(beta_rows: np.ndarray, inputs: Any, theta_sign: float) -> np.ndarray:
    beta_arr = np.asarray(beta_rows, dtype=float).reshape(-1, 6)
    xyz = np.zeros((beta_arr.shape[0], 3), dtype=float)
    for i, beta in enumerate(beta_arr):
        theta_raw = beta_to_theta(beta)
        p_xyz, _ = forward_kinematics(
            theta_raw,
            inputs.lengths_m,
            inputs.p_end_local_m,
            theta_sign=float(theta_sign),
        )
        xyz[i] = np.asarray(p_xyz, dtype=float).reshape(3)
    return xyz


def _neighbor_order(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = pts.shape[0]
    if n <= 1:
        return np.arange(n, dtype=int)
    start = int(rng.integers(0, n))
    order = [start]
    remaining = np.ones(n, dtype=bool)
    remaining[start] = False
    while len(order) < n:
        cur = order[-1]
        ridx = np.where(remaining)[0]
        d2 = np.sum((pts[ridx] - pts[cur][None, :]) ** 2, axis=1)
        nxt = int(ridx[int(np.argmin(d2))])
        order.append(nxt)
        remaining[nxt] = False
    return np.asarray(order, dtype=int)


class _TargetSampler:
    def __init__(self, rng: np.random.Generator, xyz_ranges_m: dict, mode: str, buffer_size: int) -> None:
        self.rng = rng
        self.xyz_ranges_m = xyz_ranges_m
        self.mode = mode.strip().lower()
        self.buffer_size = max(8, int(buffer_size))
        self.buffer: list[np.ndarray] = []

    def _fill_neighbor_buffer(self) -> None:
        pts = np.stack([_sample_xyz_target(self.rng, self.xyz_ranges_m) for _ in range(self.buffer_size)], axis=0)
        order = _neighbor_order(pts, self.rng)
        self.buffer = [pts[i].copy() for i in order.tolist()]

    def next(self) -> np.ndarray:
        if self.mode in {"random", ""}:
            return _sample_xyz_target(self.rng, self.xyz_ranges_m)
        if self.mode in {"neighbor_randomized", "neighbor"}:
            if not self.buffer:
                self._fill_neighbor_buffer()
            return self.buffer.pop(0)
        raise ValueError(f"Unsupported sampling.xyz_order={self.mode}")


def _write_shard(out_dir: Path, shard_idx: int, rows: list[dict], mrows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / f"dataset_part-{shard_idx:06d}.parquet"
    meta_path = out_dir / f"dataset_meta_part-{shard_idx:06d}.parquet"
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    mtable = pa.Table.from_pandas(pd.DataFrame(mrows), preserve_index=False)
    pq.write_table(table, dataset_path, compression="zstd")
    pq.write_table(mtable, meta_path, compression="zstd")


def _merge_parts(parts_dir: Path, out_path: Path, pattern: str) -> int:
    files = sorted(parts_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parts found: {pattern} in {parts_dir}")
    dataset = ds.dataset(files, format="parquet")
    pq.write_table(dataset.to_table(), out_path, compression="zstd")
    return int(dataset.count_rows())


def sort_output_tables_by_sample_id(dataset_path: Path, meta_path: Path) -> None:
    if not dataset_path.exists() or not meta_path.exists():
        return

    dataset_df = pq.read_table(dataset_path).to_pandas()
    meta_df = pq.read_table(meta_path).to_pandas()
    if "sample_id" not in dataset_df.columns or "sample_id" not in meta_df.columns:
        return

    dataset_df = dataset_df.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    meta_df = meta_df.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if dataset_df["sample_id"].tolist() != meta_df["sample_id"].tolist():
        raise SystemExit("sample_id mismatch between dataset and meta after sorting")

    pq.write_table(pa.Table.from_pandas(dataset_df, preserve_index=False), dataset_path, compression="zstd")
    pq.write_table(pa.Table.from_pandas(meta_df, preserve_index=False), meta_path, compression="zstd")


def _sample_to_forward_result(
    *,
    beta: np.ndarray,
    sample_id: int,
    sample_seed: int,
    pso_seed: int,
    model: QuasiStaticModel,
    inputs: Any,
    pso_cfg: dict[str, Any],
    rms_thresh: float,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    theta_raw = beta_to_theta(beta)
    theta = theta_raw * model.theta_sign
    p_xyz, _ = forward_kinematics(
        theta_raw,
        inputs.lengths_m,
        inputs.p_end_local_m,
        theta_sign=model.theta_sign,
    )
    cache = model.build_cache(theta_raw)
    label = solve_tension_label(model=model, cache=cache, pso_cfg=pso_cfg, pso_seed=pso_seed, rms_thresh=rms_thresh)
    T_base_12 = label.T_base_12
    meta = {**label.meta, "elapsed_s": 0.0}
    if extra_meta:
        meta.update(extra_meta)
    meta["radius_m"] = float(np.linalg.norm(p_xyz))
    sample = {
        "sample_id": int(sample_id),
        "seed": int(sample_seed),
        "beta6_rad": beta.tolist(),
        "x_m": float(p_xyz[0]),
        "y_m": float(p_xyz[1]),
        "z_m": float(p_xyz[2]),
        "theta_rad": theta.tolist(),
        "tension_n": T_base_12.tolist(),
        "meta": meta,
    }
    ok = bool(np.isfinite(p_xyz).all() and np.isfinite(T_base_12).all() and label.ok)
    return ok, sample


def _run_sequential(cfg: dict, num_samples: int, out_dir: Path, max_tried: int | None) -> None:
    inputs = load_robot_inputs(cfg)
    model = QuasiStaticModel(cfg, inputs)

    mode = str(cfg.get("dataset", {}).get("mode", "forward")).strip().lower()
    strategy = validate_generation_strategy(cfg)
    if mode not in {"forward", "inverse_joint"}:
        raise SystemExit(f"Unsupported dataset.mode={mode}, expected forward|inverse_joint")

    beta_ranges = cfg["sampling"]["beta_ranges_rad"]
    xyz_ranges_m = cfg.get("sampling", {}).get("xyz_ranges_m", {})
    rng = np.random.default_rng(int(cfg["sampling"]["rng_seed"]))
    shard_rows = int(cfg["dataset"]["shard_rows"])
    rms_thresh = float(cfg["dataset"]["rms_rnorm_threshold"])
    xyz_thresh = float(cfg["dataset"].get("xyz_err_threshold_m", float("inf")))
    pso_cfg = cfg["pso"]
    inverse_pso_cfg = cfg.get("inverse_pso", {})
    base_pso_seed = int(pso_cfg.get("rng_seed", 0))
    xyz_order = str(cfg.get("sampling", {}).get("xyz_order", "random")).strip().lower()
    xyz_neighbor_buffer = int(cfg.get("sampling", {}).get("xyz_neighbor_buffer", 256))
    target_sampler = _TargetSampler(rng=rng, xyz_ranges_m=xyz_ranges_m, mode=xyz_order, buffer_size=xyz_neighbor_buffer)
    use_cont_chain = bool(inverse_pso_cfg.get("use_continuity_chain", False))
    prev_beta: np.ndarray | None = None
    sweep_cfg = cfg.get("sampling", {}).get("standard_sweep", {})
    mixed_cfg = cfg.get("sampling", {}).get("mixed_beta", {})
    max_retries_per_candidate = max(
        1,
        int(
            mixed_cfg.get(
                "max_retries_per_candidate",
                sweep_cfg.get("max_retries_per_candidate", 5),
            )
        ),
    )
    sweep_tasks: list[dict[str, Any]] = []
    sweep_plan = None
    mixed_tasks: list[dict[str, Any]] = []
    mixed_plan = None
    if strategy == "standard_sweep":
        sweep_tasks, sweep_plan = build_standard_sweep_tasks(
            beta_ranges_rad=beta_ranges,
            num_samples=num_samples,
            axis_order=sweep_cfg.get("axis_order"),
        )
    if strategy == "mixed_beta":
        mixed_tasks, mixed_plan = build_mixed_beta_tasks(
            beta_ranges_rad=beta_ranges,
            num_samples=num_samples,
            mixed_cfg=mixed_cfg,
            rng_seed=int(cfg["sampling"]["rng_seed"]),
            workspace_xyz_fn=lambda beta_rows: _workspace_xyz_from_beta_rows(
                beta_rows,
                inputs,
                model.theta_sign,
            ),
        )

    rows: list[dict] = []
    mrows: list[dict] = []
    parts_dir = out_dir / "partials"
    parts_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 1

    accepted = 0
    tried = 0
    t_start = time.time()

    with tqdm(total=num_samples, desc="accepted") as pbar:
        last_log = time.time()
        if strategy in {"standard_sweep", "mixed_beta"}:
            planned_tasks = sweep_tasks if strategy == "standard_sweep" else mixed_tasks
            for task in planned_tasks:
                accepted_before = accepted
                for retry_idx in range(max_retries_per_candidate):
                    tried += 1
                    pso_seed = base_pso_seed + int(task["seed"]) + retry_idx
                    extra_meta = {"retry_count": retry_idx}
                    for key in [
                        "scan_axis",
                        "scan_axis_idx",
                        "scan_sign",
                        "scan_level",
                        "scan_angle_rad",
                        "scan_angle_deg",
                        "source_component",
                        "source_component_idx",
                        "beta_group1_norm",
                        "beta_group2_norm",
                        "beta_group3_norm",
                        "distal_preference_score",
                    ]:
                        if key in task:
                            extra_meta[key] = task[key]
                    ok, sample = _sample_to_forward_result(
                        beta=np.asarray(task["beta6_rad"], dtype=float).reshape(6),
                        sample_id=int(task["sample_id"]),
                        sample_seed=int(task["seed"]),
                        pso_seed=pso_seed,
                        model=model,
                        inputs=inputs,
                        pso_cfg=pso_cfg,
                        rms_thresh=rms_thresh,
                        extra_meta=extra_meta,
                    )
                    if not ok:
                        continue
                    row, mrow = _flatten_sample_rows(sample)
                    rows.append(row)
                    mrows.append(mrow)
                    accepted += 1
                    pbar.update(1)
                    if len(rows) >= shard_rows:
                        _write_shard(parts_dir, shard_idx, rows, mrows)
                        shard_idx += 1
                        rows.clear()
                        mrows.clear()
                    break
                if accepted == accepted_before:
                    raise SystemExit(
                        f"{strategy} candidate failed after retries: "
                        f"sample_id={task['sample_id']} component={task.get('source_component', task.get('scan_axis', ''))} "
                        f"level={task.get('scan_level', '')} "
                        f"max_retries_per_candidate={max_retries_per_candidate}"
                    )
                if time.time() - last_log > 10.0:
                    last_log = time.time()
                    pbar.set_postfix_str(f"tried={tried} acc_rate={accepted/max(tried,1):.3f}")
        else:
            while accepted < num_samples:
                if max_tried is not None and tried >= max_tried:
                    raise SystemExit(
                        f"Reached max_tried={max_tried} but accepted={accepted}/{num_samples}. "
                        "Likely threshold too strict or model/params inconsistent."
                    )
                tried += 1
                pso_seed = base_pso_seed + tried

                if mode == "forward":
                    beta = _sample_beta(rng, beta_ranges)
                    ok, sample = _sample_to_forward_result(
                        beta=beta,
                        sample_id=accepted,
                        sample_seed=tried,
                        pso_seed=pso_seed,
                        model=model,
                        inputs=inputs,
                        pso_cfg=pso_cfg,
                        rms_thresh=rms_thresh,
                    )
                else:
                    target_xyz = target_sampler.next()
                    res = solve_inverse_joint_pso(
                        model=model,
                        inputs=inputs,
                        xyz_target_m=target_xyz,
                        beta_ranges_rad=beta_ranges,
                        inverse_pso_cfg=inverse_pso_cfg,
                        tension_pso_cfg=pso_cfg,
                        rng_seed=pso_seed,
                        warm_start_beta=prev_beta if use_cont_chain else None,
                        continuity_ref_beta=prev_beta if use_cont_chain else None,
                    )
                    ok = bool(
                        np.isfinite(res.p_xyz_m).all()
                        and np.isfinite(res.theta_rad).all()
                        and np.isfinite(res.T_base_12).all()
                        and (res.rms_rnorm < rms_thresh)
                        and (res.xyz_err_m < xyz_thresh)
                    )
                    sample = {
                        "sample_id": accepted,
                        "seed": tried,
                        "beta6_rad": res.beta6_rad.tolist(),
                        "x_m": float(res.p_xyz_m[0]),
                        "y_m": float(res.p_xyz_m[1]),
                        "z_m": float(res.p_xyz_m[2]),
                        "theta_rad": res.theta_rad.tolist(),
                        "tension_n": res.T_base_12.tolist(),
                        "meta": {
                            "rms_rnorm": float(res.rms_rnorm),
                            "mean_rnorm2": float(res.mean_rnorm2),
                            "max_tension": float(res.max_tension),
                            "best_cost": float(res.best_cost),
                            "iters_used": int(res.iters_used),
                            "evals": int(res.evals),
                            "pso_seed": int(pso_seed),
                            "case_flag_12": res.case_flag_12.astype(int).tolist(),
                            "target_x_m": float(res.target_xyz_m[0]),
                            "target_y_m": float(res.target_xyz_m[1]),
                            "target_z_m": float(res.target_xyz_m[2]),
                            "xyz_err_m": float(res.xyz_err_m),
                            "elapsed_s": 0.0,
                        },
                    }

                if not ok:
                    if time.time() - last_log > 10.0:
                        last_log = time.time()
                        pbar.set_postfix_str(f"tried={tried} acc_rate={accepted/max(tried,1):.3f}")
                    continue

                row, mrow = _flatten_sample_rows(sample)
                rows.append(row)
                mrows.append(mrow)
                accepted += 1
                if mode == "inverse_joint" and use_cont_chain:
                    prev_beta = np.asarray(sample["beta6_rad"], dtype=float).reshape(6)
                pbar.update(1)

                if len(rows) >= shard_rows:
                    _write_shard(parts_dir, shard_idx, rows, mrows)
                    shard_idx += 1
                    rows.clear()
                    mrows.clear()

                if time.time() - last_log > 10.0:
                    last_log = time.time()
                    pbar.set_postfix_str(f"tried={tried} acc_rate={accepted/max(tried,1):.3f}")

    if rows:
        _write_shard(parts_dir, shard_idx, rows, mrows)

    t_total = time.time() - t_start
    report = {
        "mode": f"sequential:{mode}",
        "sampling_strategy": strategy,
        "accepted": accepted,
        "tried": tried,
        "accept_rate": accepted / tried,
        "elapsed_s": t_total,
        "sec_per_accepted": t_total / accepted,
    }
    if sweep_plan is not None:
        report.update(
            {
                "axis_ranges_deg": {
                    axis: [float(np.rad2deg(lo)), float(np.rad2deg(hi))]
                    for axis, (lo, hi) in sweep_plan.axis_ranges_rad.items()
                },
                "per_axis_rows": sweep_plan.per_axis_rows,
                "per_axis_positive_levels": sweep_plan.per_axis_positive_levels,
                "per_axis_step_deg": sweep_plan.per_axis_step_deg,
                "total_retries": tried - accepted,
            }
        )
    if mixed_plan is not None:
        report.update(
            {
                "axis_ranges_deg": {
                    axis: [float(np.rad2deg(lo)), float(np.rad2deg(hi))]
                    for axis, (lo, hi) in mixed_plan.axis_ranges_rad.items()
                },
                "component_counts": mixed_plan.component_counts,
                "total_retries": tried - accepted,
            }
        )
    (out_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_workers(
    cfg: dict,
    num_samples: int,
    workers: int,
    out_dir: Path,
    resume: bool,
    config_path: str,
    max_tried_tasks: int | None,
) -> None:
    mode = str(cfg.get("dataset", {}).get("mode", "forward")).strip().lower()
    strategy = validate_generation_strategy(cfg)
    if mode not in {"forward", "inverse_joint"}:
        raise SystemExit(f"Unsupported dataset.mode={mode}, expected forward|inverse_joint")

    beta_ranges = cfg["sampling"]["beta_ranges_rad"]
    xyz_ranges_m = cfg.get("sampling", {}).get("xyz_ranges_m", {})
    rng = np.random.default_rng(int(cfg["sampling"]["rng_seed"]))
    xyz_order = str(cfg.get("sampling", {}).get("xyz_order", "random")).strip().lower()
    xyz_neighbor_buffer = int(cfg.get("sampling", {}).get("xyz_neighbor_buffer", 256))
    target_sampler = _TargetSampler(rng=rng, xyz_ranges_m=xyz_ranges_m, mode=xyz_order, buffer_size=xyz_neighbor_buffer)
    shard_rows = int(cfg["dataset"]["shard_rows"])
    parts_dir = out_dir / "partials"
    parts_dir.mkdir(parents=True, exist_ok=True)
    pso_cfg = cfg["pso"]
    base_pso_seed = int(pso_cfg.get("rng_seed", 0))
    sweep_cfg = cfg.get("sampling", {}).get("standard_sweep", {})
    mixed_cfg = cfg.get("sampling", {}).get("mixed_beta", {})
    max_retries_per_candidate = max(
        1,
        int(
            mixed_cfg.get(
                "max_retries_per_candidate",
                sweep_cfg.get("max_retries_per_candidate", 5),
            )
        ),
    )
    sweep_tasks: list[dict[str, Any]] = []
    sweep_plan = None
    mixed_tasks: list[dict[str, Any]] = []
    mixed_plan = None
    if strategy in {"standard_sweep", "mixed_beta"}:
        if resume:
            raise SystemExit(f"sampling.strategy={strategy} does not support --resume")
    if strategy == "standard_sweep":
        sweep_tasks, sweep_plan = build_standard_sweep_tasks(
            beta_ranges_rad=beta_ranges,
            num_samples=num_samples,
            axis_order=sweep_cfg.get("axis_order"),
        )
    if strategy == "mixed_beta":
        inputs_for_workspace = load_robot_inputs(cfg)
        theta_sign_for_workspace = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
        mixed_tasks, mixed_plan = build_mixed_beta_tasks(
            beta_ranges_rad=beta_ranges,
            num_samples=num_samples,
            mixed_cfg=mixed_cfg,
            rng_seed=int(cfg["sampling"]["rng_seed"]),
            workspace_xyz_fn=lambda beta_rows: _workspace_xyz_from_beta_rows(
                beta_rows,
                inputs_for_workspace,
                theta_sign_for_workspace,
            ),
        )

    # resume: 统计已有 shards
    accepted = 0
    shard_idx = 1
    if resume:
        existing = sorted(parts_dir.glob("dataset_part-*.parquet"))
        if existing:
            accepted = int(sum(pq.ParquetFile(p).metadata.num_rows for p in existing))
            shard_idx = len(existing) + 1

    rows: list[dict] = []
    mrows: list[dict] = []
    tried = 0
    next_random_sample_id = accepted
    next_planned_task_idx = 0
    pending_planned: dict[int, dict[str, Any]] = {}

    cmd = [sys.executable, "scripts/worker_generate_sample.py", "--config", config_path]
    wprocs = start_workers(cmd, workers)
    time.sleep(0.2)
    dead_workers = [w for w in wprocs if w.proc.poll() is not None]
    if dead_workers:
        err_lines: list[str] = []
        for w in dead_workers:
            while not w.err_q.empty():
                err_lines.append(w.err_q.get())
        stop_workers(wprocs)
        raise SystemExit(
            "Worker processes exited during startup. "
            f"cmd={cmd}, dead={len(dead_workers)}/{workers}, "
            f"errors={(err_lines[:5] if err_lines else ['no stderr'])}"
        )

    in_flight = 0
    max_in_flight = workers * 4
    t_start = time.time()

    pbar = tqdm(total=num_samples, initial=accepted, desc="accepted")
    try:
        while accepted < num_samples:
            dead_workers = [w for w in wprocs if w.proc.poll() is not None]
            if dead_workers:
                err_lines: list[str] = []
                for w in dead_workers:
                    while not w.err_q.empty():
                        err_lines.append(w.err_q.get())
                stop_workers(wprocs)
                raise SystemExit(
                    "Worker process exited during generation. "
                    f"dead={len(dead_workers)}/{workers}, "
                    f"errors={(err_lines[:8] if err_lines else ['no stderr'])}"
                )

            # 发任务
            while in_flight < max_in_flight:
                if strategy in {"standard_sweep", "mixed_beta"}:
                    planned_tasks = sweep_tasks if strategy == "standard_sweep" else mixed_tasks
                    if next_planned_task_idx >= len(planned_tasks):
                        break
                    task = dict(planned_tasks[next_planned_task_idx])
                    task["retry_count"] = 0
                    task["pso_seed"] = base_pso_seed + int(task["seed"])
                    submit_task(wprocs[next_planned_task_idx % workers], task)
                    pending_planned[int(task["sample_id"])] = task
                    next_planned_task_idx += 1
                    tried += 1
                    in_flight += 1
                    continue

                if accepted + in_flight >= num_samples:
                    break
                if max_tried_tasks is not None and tried >= max_tried_tasks:
                    raise SystemExit(
                        f"Reached max tried tasks={max_tried_tasks} but accepted={accepted}/{num_samples}. "
                        "Likely threshold too strict or model/params inconsistent."
                    )
                tried += 1
                if mode == "forward":
                    beta = _sample_beta(rng, beta_ranges)
                    task = {"sample_id": next_random_sample_id, "seed": tried, "beta6_rad": beta.tolist()}
                else:
                    target_xyz = target_sampler.next()
                    task = {"sample_id": next_random_sample_id, "seed": tried, "target_xyz_m": target_xyz.tolist()}
                submit_task(wprocs[next_random_sample_id % workers], task)
                next_random_sample_id += 1
                in_flight += 1

            # 收结果
            res = poll_results(wprocs, timeout_s=0.5)
            if res is None:
                continue
            in_flight -= 1
            if strategy in {"standard_sweep", "mixed_beta"}:
                sample_id = int(res.get("sample_id", -1))
                task = pending_planned.pop(sample_id, None)
                if task is None:
                    raise SystemExit(f"Unknown {strategy} result sample_id={sample_id}: {res}")
                if not res.get("ok", False):
                    retry_count = int(task.get("retry_count", 0))
                    if retry_count + 1 >= max_retries_per_candidate:
                        raise SystemExit(
                            f"{strategy} candidate failed after retries: "
                            f"sample_id={sample_id} component={task.get('source_component', task.get('scan_axis'))} "
                            f"level={task.get('scan_level', '')} "
                            f"max_retries_per_candidate={max_retries_per_candidate} "
                            f"error={res.get('error', 'quality_gate_failed')}"
                        )
                    retry_task = dict(task)
                    retry_task["retry_count"] = retry_count + 1
                    retry_task["pso_seed"] = base_pso_seed + int(retry_task["seed"]) + retry_task["retry_count"]
                    submit_task(wprocs[(sample_id + retry_count + 1) % workers], retry_task)
                    pending_planned[sample_id] = retry_task
                    tried += 1
                    in_flight += 1
                    continue
            else:
                if not res.get("ok", False):
                    continue

            row, mrow = _flatten_sample_rows(res)
            rows.append(row)
            mrows.append(mrow)
            accepted += 1
            pbar.update(1)

            if len(rows) >= shard_rows:
                _write_shard(parts_dir, shard_idx, rows, mrows)
                shard_idx += 1
                rows.clear()
                mrows.clear()

        if rows:
            _write_shard(parts_dir, shard_idx, rows, mrows)
    finally:
        pbar.close()
        stop_workers(wprocs)

    t_total = time.time() - t_start
    report = {
        "mode": f"subprocess_workers:{mode}",
        "sampling_strategy": strategy,
        "workers": workers,
        "accepted": accepted,
        "tried_tasks": tried,
        "accept_rate": accepted / max(tried, 1),
        "elapsed_s": t_total,
        "sec_per_accepted": t_total / max(accepted, 1),
    }
    if sweep_plan is not None:
        report.update(
            {
                "axis_ranges_deg": {
                    axis: [float(np.rad2deg(lo)), float(np.rad2deg(hi))]
                    for axis, (lo, hi) in sweep_plan.axis_ranges_rad.items()
                },
                "per_axis_rows": sweep_plan.per_axis_rows,
                "per_axis_positive_levels": sweep_plan.per_axis_positive_levels,
                "per_axis_step_deg": sweep_plan.per_axis_step_deg,
                "total_retries": tried - accepted,
            }
        )
    if mixed_plan is not None:
        report.update(
            {
                "axis_ranges_deg": {
                    axis: [float(np.rad2deg(lo)), float(np.rad2deg(hi))]
                    for axis, (lo, hi) in mixed_plan.axis_ranges_rad.items()
                },
                "component_counts": mixed_plan.component_counts,
                "total_retries": tried - accepted,
            }
        )
    (out_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robot.yaml")
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--max-tried",
        type=int,
        default=None,
        help="最多尝试多少个采样任务（防止因阈值/参数问题无限循环）。默认不限制。",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_generation_strategy(cfg)
    out_dir = Path(cfg["dataset"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers
    if workers is None:
        workers = int(cfg["parallel"]["workers"])

    pso_backend = str(cfg.get("pso", {}).get("backend", "numpy")).strip().lower()
    inv_backend = str(cfg.get("inverse_pso", {}).get("backend", "numpy")).strip().lower()
    use_tf_backend = (
        pso_backend in {"tensorflow", "tf", "tensorflow_gpu"}
        or inv_backend in {"tensorflow", "tf", "tensorflow_gpu"}
    )
    allow_tf_multi_worker = bool(cfg.get("parallel", {}).get("allow_tf_multi_worker", False))
    use_cont_chain = bool(cfg.get("inverse_pso", {}).get("use_continuity_chain", False))
    if use_tf_backend and workers > 1 and (not allow_tf_multi_worker):
        print(
            f"[info] TensorFlow backend detected (pso={pso_backend}, inverse={inv_backend}), "
            f"forcing workers={workers} -> 1 to avoid multi-process GPU contention. "
            "Set parallel.allow_tf_multi_worker=true to override."
        )
        workers = 1
    if use_cont_chain and workers > 1:
        print(
            "[info] inverse_pso.use_continuity_chain=true requires sequential dependency, "
            f"forcing workers={workers} -> 1."
        )
        workers = 1

    if workers <= 1:
        _run_sequential(cfg, args.num_samples, out_dir, max_tried=args.max_tried)
    else:
        _run_workers(
            cfg,
            args.num_samples,
            workers=workers,
            out_dir=out_dir,
            resume=args.resume,
            config_path=args.config,
            max_tried_tasks=args.max_tried,
        )

    parts_dir = out_dir / "partials"
    n1 = _merge_parts(parts_dir, out_dir / "dataset.parquet", "dataset_part-*.parquet")
    n2 = _merge_parts(parts_dir, out_dir / "dataset_meta.parquet", "dataset_meta_part-*.parquet")
    if n1 != n2:
        raise SystemExit(f"Row mismatch after merge: dataset={n1}, meta={n2}")
    sort_output_tables_by_sample_id(out_dir / "dataset.parquet", out_dir / "dataset_meta.parquet")

    print(f"OK: wrote {n1} rows to {out_dir/'dataset.parquet'}")


if __name__ == "__main__":
    main()
