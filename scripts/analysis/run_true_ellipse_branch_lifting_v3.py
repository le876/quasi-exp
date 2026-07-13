#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from true_ellipse_atlas_utils import (  # noqa: E402
    BETA_COLS,
    TARGET_XYZ_COLS,
    THETA_COLS,
    XYZ_COLS,
    beta_bounds_rad,
    beta_rms_deg,
    branch_hash,
    branch_reproducibility_report,
    canonical_posture_penalty,
    cluster_beta_candidates,
    continuation_lift,
    evaluate_centerline_gates,
    fk_from_beta_batch,
    generate_nullspace_seeds,
    jacobian_metrics,
    link_cyclic_branch,
    link_cyclic_branch_soft,
    local_beta_consistency_report,
    make_axis_phase_ellipse,
    make_normal_tube_targets,
    numerical_jacobian_beta,
    optimize_cyclic_trajectory,
    solve_beta_ik_many,
    smoothness_report,
    theta_from_beta_batch,
    trajectory_solution_frame,
    weighted_damped_pinv,
    write_json,
    write_markdown_table,
)


DEFAULT_V2_DIR = REPO_ROOT / "runs" / "true_ellipse_reachability_atlas_v2"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "true_ellipse_branch_lifting_v3"
DEFAULT_CANDIDATE_ID = "c0156_a75_py120_pz30"


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _phase_selected(args: argparse.Namespace, name: str) -> bool:
    phases = {item.strip() for item in str(args.phases).split(",") if item.strip()}
    return "all" in phases or name in phases


def _load_robot(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    cfg = load_config(str(args.robot_config))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    return inputs.lengths_m, inputs.p_end_local_m, theta_sign


def _load_robot_from_config(config_path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    cfg = load_config(str(config_path))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    return inputs.lengths_m, inputs.p_end_local_m, theta_sign


def _v2_paths(args: argparse.Namespace) -> dict[str, Path]:
    base = Path(args.v2_dir)
    candidate_id = str(args.candidate_id)
    return {
        "candidate_table": base / "02_ellipse_family_search" / "top_candidates_by_radius.csv",
        "pointwise": base / "03_fullbeta_pointwise_ik" / candidate_id / "pointwise_candidates.parquet",
        "pointwise_report": base / "03_fullbeta_pointwise_ik" / candidate_id / "pointwise_report.json",
        "hard_link_report": base / "04_cyclic_branch_linking" / candidate_id / "linking_report.json",
        "threshold_sweep": base / "10_failure_diagnostics" / "branch_threshold_sweep.csv",
        "reachability_pool": base / "01_reachability_pool" / "reachability_pool_merged.parquet",
    }


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")


def _load_candidate_row(args: argparse.Namespace) -> pd.Series:
    path = _v2_paths(args)["candidate_table"]
    _require_file(path, "V2 ellipse candidate table")
    table = pd.read_csv(path)
    matched = table[table["candidate_id"].astype(str).eq(str(args.candidate_id))]
    if len(matched) != 1:
        raise ValueError(f"expected exactly one V2 row for {args.candidate_id}, got {len(matched)}")
    return matched.iloc[0]


def _load_v2_candidates(args: argparse.Namespace) -> pd.DataFrame:
    path = _v2_paths(args)["pointwise"]
    _require_file(path, "V2 pointwise candidates")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"V2 pointwise candidates are empty: {path}")
    return frame.sort_values(["angle_idx", "xyz_residual_mm"]).reset_index(drop=True)


def _targets_for_points(args: argparse.Namespace, n_points: int) -> pd.DataFrame:
    row = _load_candidate_row(args)
    targets = make_axis_phase_ellipse(
        candidate_id=str(args.candidate_id),
        center=(float(row["center_x_m"]), float(row["center_y_m"]), float(row["center_z_m"])),
        amp_xy_mm=float(row["amp_xy_mm"]),
        phase_y_rad=float(row["phase_y_rad"]),
        phase_z_rad=float(row["phase_z_rad"]),
        n_points=int(n_points),
    )
    return targets.sort_values("angle_idx").reset_index(drop=True)


def _hard_baseline_metrics(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected, report = link_cyclic_branch(candidates, max_edge_deg=5.0, closure_weight=2.0)
    if selected.empty:
        raise RuntimeError("V2 5-degree baseline can no longer be reconstructed")
    return selected, report


def _minimum_layer_transition_deg(left: pd.DataFrame, right: pd.DataFrame) -> float:
    a = left[BETA_COLS].to_numpy(dtype=float)
    b = right[BETA_COLS].to_numpy(dtype=float)
    diff = a[:, None, :] - b[None, :, :]
    distance = np.sqrt(np.mean(np.square(diff), axis=2)) * 180.0 / math.pi
    return float(np.min(distance))


def _gate_augmented(report: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(report)
    out.update(evaluate_centerline_gates(out))
    return out


def _report_sort_key(report: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        0.0 if bool(report.get("centerline_gate_pass", False)) else 1.0,
        0.0 if bool(report.get("branch_gate_pass", False)) else 1.0,
        float(report.get("canonical_posture_mean", np.inf)),
        float(report.get("delta_beta_p95_deg", np.inf)),
        float(report.get("residual_p95_mm", np.inf)),
    )


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_augmented_report(path: Path) -> dict[str, Any]:
    return _gate_augmented(_safe_read_json(path))


def _selected_branch_robustness_gate(
    *,
    selected_pair: Mapping[str, Any],
    repeatability: Mapping[str, Any],
    selected_report: Mapping[str, Any],
) -> bool:
    return bool(
        selected_pair.get("reproducible", False)
        and repeatability.get("reproducible", False)
        and selected_report.get("centerline_gate_pass", False)
    )


def _load_robustness_report(path: Path) -> dict[str, Any]:
    report = dict(_safe_read_json(path))
    selected_report = _gate_augmented(report.get("selected_run", {}))
    report["selected_run"] = selected_report
    report["robustness_gate_pass"] = _selected_branch_robustness_gate(
        selected_pair=report.get("selected_forward_reverse", {}),
        repeatability=report.get("deterministic_repeatability", {}),
        selected_report=selected_report,
    )
    return report


def _execute_worker_tasks(task_paths: Iterable[Path], *, workers: int) -> None:
    paths = [Path(path) for path in task_paths]
    if not paths:
        return

    def execute(task_path: Path) -> None:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-task", str(task_path)],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"worker failed for {task_path} (exit {result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        list(executor.map(execute, paths))


def run_worker_task(task_path: Path) -> None:
    task = _safe_read_json(Path(task_path))
    kind = str(task.get("kind", ""))
    lengths_m, p_end_local_m, theta_sign = _load_robot_from_config(task["robot_config"])
    bounds = beta_bounds_rad("current")
    if kind == "continuation":
        targets = pd.read_parquet(task["targets_path"])
        lifted, report = continuation_lift(
            targets=targets,
            start_beta=np.asarray(task["start_beta"], dtype=float),
            bounds=bounds,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
            direction=str(task["direction"]),
            method=str(task["method"]),
            lambda_center=float(task["center_lambda"]),
            lambda_limit=0.0,
            max_nfev=int(task["max_ik_nfev"]),
        )
        report = _gate_augmented(report)
        report.update(
            {
                "run_id": str(task["run_id"]),
                "start_candidate_idx": int(task["start_idx"]),
                "start_branch_cluster_id": int(task["start_branch_cluster_id"]),
                "path": str(task["path_file"]),
                "branch_hash": branch_hash(lifted),
            }
        )
        report.update(dict(task.get("report_metadata", {})))
        lifted["continuation_run_id"] = str(task["run_id"])
        lifted["start_candidate_idx"] = int(task["start_idx"])
        lifted.to_parquet(task["path_file"], index=False, compression="zstd")
        write_json(Path(task["report_file"]), report)
        return
    if kind == "enrich_angle":
        target_row = pd.Series(task["target_row"])
        seeds = [
            {"source": str(source), "beta": np.asarray(beta, dtype=float), "seed_order": int(index)}
            for index, (source, beta) in enumerate(zip(task["seed_sources"], task["seed_betas"]))
        ]
        base_budget = min(int(task["base_budget"]), len(seeds))
        adaptive_budget = min(int(task["adaptive_budget"]), len(seeds))
        target_xyz = target_row[TARGET_XYZ_COLS].to_numpy(dtype=float)

        def solve_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not records:
                return []
            solutions = solve_beta_ik_many(
                target_xyz,
                init_betas=np.vstack([record["beta"] for record in records]),
                bounds=bounds,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
                max_nfev=int(task["max_ik_nfev"]),
                lambda_limit=0.0,
                lambda_center=1.0e-4,
                center_each_seed=True,
            )
            return [
                _solution_candidate_record(
                    target_row,
                    solution,
                    records[solution.seed_rank],
                    lengths_m=lengths_m,
                    p_end_local_m=p_end_local_m,
                    theta_sign=theta_sign,
                )
                for solution in solutions
                if solution.residual_mm <= 5.0
            ]

        candidate_records = solve_records(seeds[:base_budget])
        initial_df = pd.DataFrame(candidate_records)
        clustered = cluster_beta_candidates(
            initial_df,
            beta_rms_threshold_deg=0.25,
            max_clusters=adaptive_budget,
        ) if not initial_df.empty else initial_df
        adaptive_trigger = bool(len(clustered) < 8 or bool(task.get("adaptive_forced", False)))
        if adaptive_trigger and adaptive_budget > base_budget:
            candidate_records.extend(solve_records(seeds[base_budget:adaptive_budget]))
        raw = pd.DataFrame(candidate_records)
        raw["adaptive_trigger"] = adaptive_trigger
        raw.to_parquet(task["raw_output"], index=False, compression="zstd")
        write_json(
            Path(task["report_output"]),
            {
                "angle_idx": int(task["angle_idx"]),
                "seed_inventory_count": int(len(seeds)),
                "base_seed_budget": int(base_budget),
                "adaptive_seed_budget": int(adaptive_budget),
                "adaptive_trigger": adaptive_trigger,
                "raw_candidate_rows": int(len(raw)),
            },
        )
        return
    if kind == "tube_curve":
        centerline = pd.read_parquet(task["centerline_path"]).sort_values("angle_idx").reset_index(drop=True)
        curve_targets = pd.read_parquet(task["curve_targets_path"]).sort_values("angle_idx").reset_index(drop=True)
        parent_curve = pd.read_parquet(task["parent_curve_path"]).sort_values("angle_idx").reset_index(drop=True)
        pinv = np.load(task["pinv_path"])
        center_beta = centerline[BETA_COLS].to_numpy(dtype=float)
        center_xyz = centerline[XYZ_COLS].to_numpy(dtype=float)
        rows: list[dict[str, Any]] = []
        previous_beta: np.ndarray | None = None
        for angle_idx, target_row in curve_targets.iterrows():
            target_xyz = target_row[TARGET_XYZ_COLS].to_numpy(dtype=float)
            prediction = np.clip(
                center_beta[int(angle_idx)] + pinv[int(angle_idx)] @ (target_xyz - center_xyz[int(angle_idx)]),
                bounds[:, 0],
                bounds[:, 1],
            )
            parent_beta = parent_curve.iloc[int(angle_idx)][BETA_COLS].to_numpy(dtype=float)
            seeds = [prediction, parent_beta, center_beta[int(angle_idx)]]
            if previous_beta is not None:
                seeds.insert(0, previous_beta)
            solutions = solve_beta_ik_many(
                target_xyz,
                init_betas=np.vstack(seeds),
                bounds=bounds,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
                max_nfev=int(task["max_ik_nfev"]),
                lambda_limit=0.0,
                center_beta=prediction,
                lambda_center=1.0e-3,
            )
            feasible = [solution for solution in solutions if solution.residual_mm <= 1.5]
            reference_beta = previous_beta if previous_beta is not None else parent_beta
            chosen = min(
                feasible if feasible else solutions,
                key=lambda solution: (
                    beta_rms_deg(solution.beta_rad, reference_beta)
                    + 0.25 * beta_rms_deg(solution.beta_rad, parent_beta)
                    + 0.10 * beta_rms_deg(solution.beta_rad, prediction),
                    solution.residual_mm,
                ),
            )
            previous_beta = chosen.beta_rad
            record = target_row.to_dict()
            record.update(
                {
                    "xyz_residual_mm": float(chosen.residual_mm),
                    "tube_success": bool(chosen.residual_mm <= 1.5),
                    "parent_offset_id": str(task["parent_offset_id"]),
                    "inverse_nfev": int(chosen.nfev),
                }
            )
            for j, col in enumerate(BETA_COLS):
                record[col] = float(chosen.beta_rad[j])
            theta = theta_from_beta_batch(chosen.beta_rad.reshape(1, 6), theta_sign=theta_sign)[0]
            for j, col in enumerate(THETA_COLS):
                record[col] = float(theta[j])
            for j, col in enumerate(XYZ_COLS):
                record[col] = float(chosen.xyz_m[j])
            rows.append(record)
        curve = pd.DataFrame(rows).sort_values("angle_idx").reset_index(drop=True)
        curve.to_parquet(task["curve_output"], index=False, compression="zstd")
        write_json(
            Path(task["report_output"]),
            {
                "offset_id": str(task["offset_id"]),
                "rows": int(len(curve)),
                "success_ratio": float(curve["tube_success"].mean()),
                "residual_p95_mm": float(np.percentile(curve["xyz_residual_mm"], 95)),
                "residual_max_mm": float(curve["xyz_residual_mm"].max()),
                **smoothness_report(curve),
            },
        )
        return
    raise ValueError(f"unsupported worker task kind: {kind}")


def apply_preset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "smoke":
        args.coarse_points = min(int(args.coarse_points), 12)
        args.final_points = min(int(args.final_points), 24)
        args.max_ik_nfev = min(int(args.max_ik_nfev), 30)
        args.max_opt_nfev = min(int(args.max_opt_nfev), 8)
        args.robustness_random_seeds = min(int(args.robustness_random_seeds), 1)
        args.workers = min(int(args.workers), 2)
    elif args.preset == "pilot":
        args.coarse_points = int(args.coarse_points)
        args.final_points = int(args.final_points)
    elif args.preset == "formal":
        args.coarse_points = max(int(args.coarse_points), 72)
        args.final_points = max(int(args.final_points), 360)
        args.max_ik_nfev = max(int(args.max_ik_nfev), 100)
        args.max_opt_nfev = max(int(args.max_opt_nfev), 40)
    else:
        raise ValueError(f"unsupported preset: {args.preset}")
    return args


def phase_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir) / "00_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _v2_paths(args)
    for label, path in paths.items():
        if label == "reachability_pool":
            continue
        _require_file(path, label)
    candidates = _load_v2_candidates(args)
    if not candidates["candidate_id"].astype(str).eq(str(args.candidate_id)).all():
        raise ValueError("V2 pointwise file contains another candidate_id")
    angle_values = sorted(candidates["angle_idx"].astype(int).unique().tolist())
    expected_angles = list(range(len(angle_values)))
    if angle_values != expected_angles:
        raise ValueError("V2 angle_idx is not contiguous from zero")
    if len(angle_values) != 72:
        raise ValueError(f"V2 audit expects 72 layers, got {len(angle_values)}")
    target_spread = candidates.groupby("angle_idx")[TARGET_XYZ_COLS].agg(np.ptp).to_numpy(dtype=float)
    if float(np.max(target_spread)) > 1.0e-12:
        raise ValueError("V2 candidates disagree on target xyz within an angle layer")
    if not np.isfinite(candidates[[*BETA_COLS, *TARGET_XYZ_COLS, "xyz_residual_mm"]].to_numpy(dtype=float)).all():
        raise ValueError("V2 candidates contain non-finite beta/target/residual values")

    counts = candidates.groupby("angle_idx").size().rename("distinct_candidate_count").reset_index()
    counts["sparse_le2"] = counts["distinct_candidate_count"].le(2)
    transition_rows: list[dict[str, Any]] = []
    layers = {int(idx): part for idx, part in candidates.groupby("angle_idx", sort=True)}
    for angle_idx in angle_values:
        next_idx = (int(angle_idx) + 1) % len(angle_values)
        transition_rows.append(
            {
                "angle_idx": int(angle_idx),
                "next_angle_idx": int(next_idx),
                "minimum_candidate_transition_deg": _minimum_layer_transition_deg(layers[int(angle_idx)], layers[int(next_idx)]),
            }
        )
    transitions = pd.DataFrame(transition_rows)
    counts = counts.merge(transitions[["angle_idx", "minimum_candidate_transition_deg"]], on="angle_idx", how="left")
    counts["bottleneck_gt1p5"] = counts["minimum_candidate_transition_deg"].gt(1.5)
    counts.to_csv(out_dir / "candidate_count_by_angle.csv", index=False)
    transitions.to_csv(out_dir / "baseline_transition_by_angle.csv", index=False)

    baseline, baseline_report = _hard_baseline_metrics(candidates)
    baseline.to_parquet(out_dir / "baseline_linked_5deg.parquet", index=False, compression="zstd")
    baseline_report = _gate_augmented(baseline_report)
    threshold = pd.read_csv(paths["threshold_sweep"])
    threshold_row = threshold[
        threshold["candidate_id"].astype(str).eq(str(args.candidate_id))
        & np.isclose(threshold["threshold_deg"].astype(float), 5.0)
    ]
    if len(threshold_row) != 1:
        raise ValueError("V2 threshold sweep is missing the unique 5-degree baseline row")
    recorded = threshold_row.iloc[0]
    for report_key, recorded_key in (
        ("delta_beta_rms_p95_deg", "delta_beta_rms_p95_deg"),
        ("delta_beta_rms_max_deg", "delta_beta_rms_max_deg"),
        ("seam_beta_rms_deg", "seam_beta_rms_deg"),
    ):
        if not math.isclose(float(baseline_report[report_key]), float(recorded[recorded_key]), rel_tol=1.0e-8, abs_tol=1.0e-8):
            raise ValueError(f"reconstructed V2 baseline disagrees for {report_key}")

    summary = {
        "candidate_id": str(args.candidate_id),
        "pointwise_rows": int(len(candidates)),
        "angle_layers": int(len(angle_values)),
        "candidate_count_min": int(counts["distinct_candidate_count"].min()),
        "candidate_count_p50": float(counts["distinct_candidate_count"].median()),
        "candidate_count_p95": float(np.percentile(counts["distinct_candidate_count"], 95)),
        "sparse_le2_layers": int(counts["sparse_le2"].sum()),
        "bottleneck_gt1p5_layers": int(counts["bottleneck_gt1p5"].sum()),
        "v2_baseline": baseline_report,
        "audit_pass": True,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    write_json(out_dir / "audit_report.json", summary)
    return summary


def phase_continuation(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "01_continuation_ablation"
    paths_dir = out_dir / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)
    audit_path = Path(args.out_dir) / "00_audit" / "audit_report.json"
    if not audit_path.exists():
        phase_audit(args)
    audit = _safe_read_json(audit_path)
    if not bool(audit.get("audit_pass", False)):
        raise RuntimeError("V2 audit did not pass")
    targets = _targets_for_points(args, int(args.coarse_points))
    targets_path = out_dir / "continuation_targets.parquet"
    targets.to_parquet(targets_path, index=False, compression="zstd")
    task_dir = out_dir / "worker_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    v2 = _load_v2_candidates(args)
    starts = v2[v2["angle_idx"].astype(int).eq(0)].sort_values("xyz_residual_mm").reset_index(drop=True)
    if args.preset == "smoke":
        starts = starts.head(1)
        lambdas = [1.0e-3]
    else:
        lambdas = _parse_float_csv(args.center_lambdas)
    if not lambdas:
        raise ValueError("center-lambdas must not be empty")
    methods = ["warm", "predictive"]
    directions = ["forward", "reverse"]
    reports: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for start_idx, start_row in starts.iterrows():
        start_beta = start_row[BETA_COLS].to_numpy(dtype=float)
        for method in methods:
            for direction in directions:
                for center_lambda in lambdas:
                    lambda_id = f"{center_lambda:.4g}".replace(".", "p").replace("-", "m")
                    run_id = f"start{int(start_idx):02d}_{method}_{direction}_lc{lambda_id}"
                    path_file = paths_dir / f"{run_id}.parquet"
                    report_file = paths_dir / f"{run_id}.json"
                    if bool(args.skip_existing) and path_file.exists() and report_file.exists():
                        report = _load_augmented_report(report_file)
                        reports.append(report)
                        continue
                    pending.append(
                        {
                            "start_idx": int(start_idx),
                            "start_branch_cluster_id": int(start_row.get("branch_cluster_id", start_idx)),
                            "start_beta": start_beta.copy(),
                            "method": method,
                            "direction": direction,
                            "center_lambda": float(center_lambda),
                            "run_id": run_id,
                            "path_file": path_file,
                            "report_file": report_file,
                        }
                    )

    worker_tasks: list[Path] = []
    if pending:
        for case in pending:
            task_path = task_dir / f"{case['run_id']}.json"
            write_json(
                task_path,
                {
                    "kind": "continuation",
                    "robot_config": str(args.robot_config),
                    "targets_path": str(targets_path),
                    "max_ik_nfev": int(args.max_ik_nfev),
                    **case,
                },
            )
            worker_tasks.append(task_path)
        _execute_worker_tasks(worker_tasks, workers=int(args.workers))
        reports.extend(_load_augmented_report(Path(case["report_file"])) for case in pending)

    summary = pd.DataFrame(reports)
    summary.to_csv(out_dir / "continuation_ablation_summary.csv", index=False)
    if summary.empty:
        write_json(out_dir / "selection_report.json", {"success": False, "reason": "no_continuation_runs"})
        return summary
    ranked = sorted(reports, key=_report_sort_key)
    selected_report = ranked[0]
    selected_path = pd.read_parquet(Path(selected_report["path"]))
    selected_path.to_parquet(out_dir / "best_continuation.parquet", index=False, compression="zstd")
    selection = {
        "success": bool(selected_report.get("branch_gate_pass", False)),
        "selected_run_id": str(selected_report["run_id"]),
        "selected_report": selected_report,
        "passing_branch_runs": int(summary.get("branch_gate_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "passing_centerline_runs": int(summary.get("centerline_gate_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "total_runs": int(len(summary)),
    }
    write_json(out_dir / "selection_report.json", selection)
    return summary


def _deduplicate_seed_records(records: list[dict[str, Any]], *, threshold_deg: float = 1.0e-5) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for record in records:
        beta = np.asarray(record["beta"], dtype=float).reshape(6)
        if any(beta_rms_deg(beta, np.asarray(old["beta"], dtype=float)) <= float(threshold_deg) for old in kept):
            continue
        kept.append({**record, "beta": beta})
    for seed_order, record in enumerate(kept):
        record["seed_order"] = int(seed_order)
    return kept


def _solution_candidate_record(
    target_row: pd.Series,
    solution: Any,
    seed_record: Mapping[str, Any],
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
) -> dict[str, Any]:
    record = target_row.to_dict()
    record.update(
        {
            "xyz_residual_mm": float(solution.residual_mm),
            "ik_success": bool(solution.residual_mm <= 2.0),
            "inverse_nfev": int(solution.nfev),
            "seed_order": int(seed_record["seed_order"]),
            "seed_source": str(seed_record["source"]),
        }
    )
    for j, col in enumerate(BETA_COLS):
        record[col] = float(solution.beta_rad[j])
    theta = theta_from_beta_batch(solution.beta_rad.reshape(1, 6), theta_sign=theta_sign)[0]
    for j, col in enumerate(THETA_COLS):
        record[col] = float(theta[j])
    for j, col in enumerate(XYZ_COLS):
        record[col] = float(solution.xyz_m[j])
    jac = numerical_jacobian_beta(
        solution.beta_rad,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
    )
    record.update(jacobian_metrics(jac))
    record["canonical_posture_cost"] = canonical_posture_penalty(solution.beta_rad)
    return record


def _continuation_paths_for_enrichment(args: argparse.Namespace) -> list[pd.DataFrame]:
    summary_path = Path(args.out_dir) / "01_continuation_ablation" / "continuation_ablation_summary.csv"
    if not summary_path.exists():
        phase_continuation(args)
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return []
    if "branch_gate_pass" in summary.columns:
        passing = summary[summary["branch_gate_pass"].astype(bool)]
        if not passing.empty:
            summary = passing
    paths: list[pd.DataFrame] = []
    for path_value in summary["path"].head(16):
        path = Path(str(path_value))
        if path.exists():
            paths.append(pd.read_parquet(path).sort_values("angle_idx").reset_index(drop=True))
    return paths


def phase_enrich(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "02_candidate_enrichment"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_file = out_dir / "candidates_adaptive.parquet"
    summary_file = out_dir / "candidate_density_summary.csv"
    if bool(args.skip_existing) and final_file.exists() and summary_file.exists():
        return pd.read_csv(summary_file)
    best_path = Path(args.out_dir) / "01_continuation_ablation" / "best_continuation.parquet"
    if not best_path.exists():
        phase_continuation(args)
    selection = _safe_read_json(Path(args.out_dir) / "01_continuation_ablation" / "selection_report.json")
    if not bool(selection.get("success", False)):
        write_json(out_dir / "enrichment_report.json", {"success": False, "reason": "no_passing_continuation"})
        return pd.DataFrame()

    lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    bounds = beta_bounds_rad("current")
    targets = _targets_for_points(args, int(args.coarse_points))
    reference = pd.read_parquet(best_path).sort_values("angle_idx").reset_index(drop=True)
    v2 = _load_v2_candidates(args)
    continuation_paths = _continuation_paths_for_enrichment(args)
    audit_counts = pd.read_csv(Path(args.out_dir) / "00_audit" / "candidate_count_by_angle.csv")

    pool_path = _v2_paths(args)["reachability_pool"]
    _require_file(pool_path, "V2 reachability pool")
    pool = pd.read_parquet(pool_path, columns=[*XYZ_COLS, *BETA_COLS])
    pool_nn = NearestNeighbors(n_neighbors=min(int(args.adaptive_seed_budget), len(pool))).fit(pool[XYZ_COLS].to_numpy(dtype=float))
    requested_budgets = sorted(set([4, *_parse_int_csv(args.seed_budgets)]))
    base_budget = max(requested_budgets)
    adaptive_budget = max(base_budget, int(args.adaptive_seed_budget))
    if args.preset == "smoke":
        base_budget = min(base_budget, 8)
        adaptive_budget = min(adaptive_budget, 12)

    raw_by_angle: dict[int, pd.DataFrame] = {}
    seed_inventory_rows: list[dict[str, Any]] = []
    v2_layer_count = int(v2["angle_idx"].nunique())
    rng_seed = int(args.seed)
    _pool_distance, pool_indices_all = pool_nn.kneighbors(
        targets[TARGET_XYZ_COLS].to_numpy(dtype=float),
        n_neighbors=min(adaptive_budget, len(pool)),
    )

    worker_task_dir = out_dir / "worker_tasks"
    raw_dir = out_dir / "raw_by_angle"
    worker_task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    worker_tasks: list[Path] = []
    worker_outputs: list[tuple[int, Path, Path]] = []
    for angle_idx, target_row in targets.iterrows():
        reference_beta = reference.iloc[int(angle_idx)][BETA_COLS].to_numpy(dtype=float)
        v2_idx = int(round(int(angle_idx) * v2_layer_count / len(targets))) % v2_layer_count
        records: list[dict[str, Any]] = []
        for _, row in v2[v2["angle_idx"].astype(int).eq(v2_idx)].iterrows():
            records.append({"source": "v2_candidate", "beta": row[BETA_COLS].to_numpy(dtype=float)})
        for path_idx, path in enumerate(continuation_paths):
            mapped_idx = int(round(int(angle_idx) * len(path) / len(targets))) % len(path)
            records.append({"source": f"continuation_{path_idx:02d}", "beta": path.iloc[mapped_idx][BETA_COLS].to_numpy(dtype=float)})
        prev_beta = reference.iloc[(int(angle_idx) - 1) % len(reference)][BETA_COLS].to_numpy(dtype=float)
        next_beta = reference.iloc[(int(angle_idx) + 1) % len(reference)][BETA_COLS].to_numpy(dtype=float)
        records.extend(
            [
                {"source": "reference_current", "beta": reference_beta},
                {"source": "neighbor_previous", "beta": prev_beta},
                {"source": "neighbor_next", "beta": next_beta},
                {"source": "linear_prediction", "beta": np.clip(2.0 * reference_beta - prev_beta, bounds[:, 0], bounds[:, 1])},
            ]
        )
        target_xyz = target_row[TARGET_XYZ_COLS].to_numpy(dtype=float)
        for rank, pool_row in enumerate(pool_indices_all[int(angle_idx)]):
            records.append({"source": f"reachability_nn_{rank:02d}", "beta": pool.iloc[int(pool_row)][BETA_COLS].to_numpy(dtype=float)})
        jac = numerical_jacobian_beta(
            reference_beta,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
        )
        null_seeds = generate_nullspace_seeds(
            reference_beta,
            jac,
            bounds=bounds,
            scales_deg=[0.25, 0.5, 1.0, 2.0],
            seeds_per_scale=6,
            seed=rng_seed + int(angle_idx),
        )
        for rank, beta in enumerate(null_seeds):
            records.append({"source": f"nullspace_{rank:02d}", "beta": beta})
        seeds = _deduplicate_seed_records(records)
        audit_row = audit_counts.iloc[v2_idx]
        raw_output = raw_dir / f"angle_{int(angle_idx):03d}.parquet"
        report_output = raw_dir / f"angle_{int(angle_idx):03d}.json"
        worker_outputs.append((int(angle_idx), raw_output, report_output))
        if bool(args.skip_existing) and raw_output.exists() and report_output.exists():
            continue
        task_path = worker_task_dir / f"enrich_angle_{int(angle_idx):03d}.json"
        write_json(
            task_path,
            {
                "kind": "enrich_angle",
                "robot_config": str(args.robot_config),
                "angle_idx": int(angle_idx),
                "target_row": target_row.to_dict(),
                "seed_sources": [str(record["source"]) for record in seeds],
                "seed_betas": [np.asarray(record["beta"], dtype=float).tolist() for record in seeds],
                "base_budget": int(base_budget),
                "adaptive_budget": int(adaptive_budget),
                "adaptive_forced": bool(
                    bool(audit_row.get("sparse_le2", False))
                    or bool(audit_row.get("bottleneck_gt1p5", False))
                ),
                "max_ik_nfev": int(args.max_ik_nfev),
                "raw_output": str(raw_output),
                "report_output": str(report_output),
            },
        )
        worker_tasks.append(task_path)
    _execute_worker_tasks(worker_tasks, workers=int(args.workers))
    for angle_idx, raw_output, report_output in worker_outputs:
        _require_file(raw_output, f"enrichment raw angle {angle_idx}")
        _require_file(report_output, f"enrichment report angle {angle_idx}")
        raw_by_angle[int(angle_idx)] = pd.read_parquet(raw_output)
        seed_inventory_rows.append(_safe_read_json(report_output))
    raw_by_angle = dict(sorted(raw_by_angle.items()))
    seed_inventory_rows.sort(key=lambda row: int(row["angle_idx"]))

    pd.DataFrame(seed_inventory_rows).to_csv(out_dir / "seed_inventory_by_angle.csv", index=False)
    budget_frames: dict[str, pd.DataFrame] = {}
    density_rows: list[dict[str, Any]] = []
    for requested_budget in requested_budgets:
        effective_budget = min(int(requested_budget), base_budget)
        clustered_layers: list[pd.DataFrame] = []
        counts: list[int] = []
        for angle_idx, raw in raw_by_angle.items():
            subset = raw[raw["seed_order"].astype(int).lt(effective_budget)].copy()
            clustered = cluster_beta_candidates(subset, beta_rms_threshold_deg=0.25, max_clusters=effective_budget) if not subset.empty else subset
            clustered["requested_seed_budget"] = int(requested_budget)
            clustered["effective_seed_budget"] = int(effective_budget)
            clustered_layers.append(clustered)
            counts.append(int(len(clustered)))
        combined = pd.concat(clustered_layers, ignore_index=True) if clustered_layers else pd.DataFrame()
        label = f"budget_{int(requested_budget)}"
        combined.to_parquet(out_dir / f"candidates_{label}.parquet", index=False, compression="zstd")
        budget_frames[label] = combined
        density_rows.append(
            {
                "candidate_set": label,
                "requested_seed_budget": int(requested_budget),
                "effective_seed_budget": int(effective_budget),
                "distinct_candidate_min": int(np.min(counts)),
                "distinct_candidate_p50": float(np.percentile(counts, 50)),
                "distinct_candidate_p95": float(np.percentile(counts, 95)),
                "distinct_candidate_max": int(np.max(counts)),
            }
        )

    adaptive_layers: list[pd.DataFrame] = []
    adaptive_counts: list[int] = []
    for raw in raw_by_angle.values():
        clustered = cluster_beta_candidates(raw, beta_rms_threshold_deg=0.25, max_clusters=adaptive_budget) if not raw.empty else raw
        clustered["requested_seed_budget"] = int(args.adaptive_seed_budget)
        clustered["effective_seed_budget"] = int(adaptive_budget)
        adaptive_layers.append(clustered)
        adaptive_counts.append(int(len(clustered)))
    adaptive = pd.concat(adaptive_layers, ignore_index=True) if adaptive_layers else pd.DataFrame()
    adaptive.to_parquet(final_file, index=False, compression="zstd")
    density_rows.append(
        {
            "candidate_set": "adaptive",
            "requested_seed_budget": int(args.adaptive_seed_budget),
            "effective_seed_budget": int(adaptive_budget),
            "distinct_candidate_min": int(np.min(adaptive_counts)),
            "distinct_candidate_p50": float(np.percentile(adaptive_counts, 50)),
            "distinct_candidate_p95": float(np.percentile(adaptive_counts, 95)),
            "distinct_candidate_max": int(np.max(adaptive_counts)),
        }
    )
    density = pd.DataFrame(density_rows)
    density.to_csv(summary_file, index=False)
    write_json(
        out_dir / "enrichment_report.json",
        {
            "success": bool(len(adaptive) > 0),
            "candidate_sets": density.to_dict("records"),
            "adaptive_trigger_layers": int(sum(bool(raw["adaptive_trigger"].iloc[0]) for raw in raw_by_angle.values() if not raw.empty)),
            "cluster_threshold_beta_rms_deg": 0.25,
            "note": "seed budgets are optimization starts; distinct candidate counts are reported separately",
        },
    )
    return density


def phase_link(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "03_soft_cyclic_linking"
    paths_dir = out_dir / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)
    enrichment_summary = Path(args.out_dir) / "02_candidate_enrichment" / "candidate_density_summary.csv"
    if not enrichment_summary.exists():
        phase_enrich(args)
    candidate_files = sorted((Path(args.out_dir) / "02_candidate_enrichment").glob("candidates_*.parquet"))
    if not candidate_files:
        write_json(out_dir / "selection_report.json", {"success": False, "reason": "no_enriched_candidates"})
        return pd.DataFrame()
    lambda_values = [0.5, 1.0, 2.0, 5.0]
    closure_values = [2.0, 5.0, 10.0]
    if args.preset == "smoke":
        lambda_values = [1.0]
        closure_values = [5.0]
    reports: list[dict[str, Any]] = []
    for candidate_file in candidate_files:
        candidate_set = candidate_file.stem.removeprefix("candidates_")
        candidates = pd.read_parquet(candidate_file)
        if candidates.empty:
            continue
        for lambda_velocity in lambda_values:
            for closure_weight in closure_values:
                run_id = f"{candidate_set}_lv{lambda_velocity:g}_cw{closure_weight:g}".replace(".", "p")
                path_file = paths_dir / f"{run_id}.parquet"
                report_file = paths_dir / f"{run_id}.json"
                if bool(args.skip_existing) and path_file.exists() and report_file.exists():
                    reports.append(_load_augmented_report(report_file))
                    continue
                selected, report = link_cyclic_branch_soft(
                    candidates,
                    lambda_velocity=lambda_velocity,
                    closure_weight=closure_weight,
                    lambda_posture=0.05,
                    lambda_kappa=0.01,
                )
                report = _gate_augmented(report)
                report.update(
                    {
                        "run_id": run_id,
                        "candidate_set": candidate_set,
                        "candidate_file": str(candidate_file),
                        "path": str(path_file),
                    }
                )
                if not selected.empty:
                    selected["soft_link_run_id"] = run_id
                    selected.to_parquet(path_file, index=False, compression="zstd")
                    report["branch_hash"] = branch_hash(selected)
                write_json(report_file, report)
                reports.append(report)
    summary = pd.DataFrame(reports)
    summary.to_csv(out_dir / "soft_linking_summary.csv", index=False)
    successful = [report for report in reports if bool(report.get("success", False)) and Path(str(report.get("path", ""))).exists()]
    if not successful:
        write_json(out_dir / "selection_report.json", {"success": False, "reason": "no_soft_cyclic_path"})
        return summary
    selected_report = sorted(successful, key=_report_sort_key)[0]
    selected = pd.read_parquet(Path(selected_report["path"]))
    selected.to_parquet(out_dir / "best_soft_linked.parquet", index=False, compression="zstd")
    write_json(
        out_dir / "selection_report.json",
        {
            "success": bool(selected_report.get("branch_gate_pass", False)),
            "selected_run_id": str(selected_report["run_id"]),
            "selected_report": selected_report,
            "total_runs": int(len(summary)),
        },
    )
    return summary


def _write_path_report(path: pd.DataFrame, report: Mapping[str, Any], *, path_file: Path, report_file: Path) -> None:
    path_file.parent.mkdir(parents=True, exist_ok=True)
    path.to_parquet(path_file, index=False, compression="zstd")
    write_json(report_file, dict(report))


def phase_optimize(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "04_trajectory_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    continuation_path = Path(args.out_dir) / "01_continuation_ablation" / "best_continuation.parquet"
    if not continuation_path.exists():
        phase_continuation(args)
    if not continuation_path.exists():
        write_json(out_dir / "optimization_report.json", {"success": False, "reason": "no_continuation_path"})
        return pd.DataFrame()
    soft_path = Path(args.out_dir) / "03_soft_cyclic_linking" / "best_soft_linked.parquet"
    if not soft_path.exists() and _phase_selected(args, "link"):
        phase_link(args)

    lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    bounds = beta_bounds_rad("current")
    sources: list[tuple[str, Path]] = [("continuation", continuation_path)]
    if soft_path.exists():
        sources.append(("soft_dp", soft_path))
    reports: list[dict[str, Any]] = []
    path_by_id: dict[str, pd.DataFrame] = {}
    for source_name, source_path in sources:
        initial = pd.read_parquet(source_path).sort_values("angle_idx").reset_index(drop=True)
        targets = _targets_for_points(args, len(initial))
        optimized, report = optimize_cyclic_trajectory(
            targets=targets,
            initial_beta=initial[BETA_COLS].to_numpy(dtype=float),
            bounds=bounds,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
            max_nfev=int(args.max_opt_nfev),
        )
        report = _gate_augmented(report)
        report.update({"path_id": f"{source_name}_optimized_72", "source": source_name, "resolution_points": int(len(optimized))})
        path_file = out_dir / f"{source_name}_optimized_72.parquet"
        _write_path_report(optimized, report, path_file=path_file, report_file=out_dir / f"{source_name}_optimized_72.json")
        report["path"] = str(path_file)
        reports.append(report)
        path_by_id[str(report["path_id"])] = optimized

    passing_72 = [report for report in reports if bool(report.get("centerline_gate_pass", False))]
    if not passing_72:
        summary = pd.DataFrame(reports)
        summary.to_csv(out_dir / "optimization_summary.csv", index=False)
        write_json(
            out_dir / "optimization_report.json",
            {"success": False, "reason": "no_passing_72_centerline", "reports": reports},
        )
        return summary
    selected_72_report = sorted(passing_72, key=_report_sort_key)[0]
    selected_72 = pd.read_parquet(Path(selected_72_report["path"]))
    selected_72.to_parquet(out_dir / "selected_centerline_72.parquet", index=False, compression="zstd")

    targets_360 = _targets_for_points(args, int(args.final_points))
    start_beta = selected_72.sort_values("angle_idx").iloc[0][BETA_COLS].to_numpy(dtype=float)
    refined, refined_report = continuation_lift(
        targets=targets_360,
        start_beta=start_beta,
        bounds=bounds,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        direction="forward",
        method="warm",
        lambda_center=1.0e-3,
        lambda_limit=0.0,
        max_nfev=int(args.max_ik_nfev),
    )
    refined_report = _gate_augmented(refined_report)
    refined_report.update({"path_id": "continuation_preopt_360", "source": "refined_continuation", "resolution_points": int(len(refined))})
    refined_file = out_dir / "continuation_preopt_360.parquet"
    _write_path_report(refined, refined_report, path_file=refined_file, report_file=out_dir / "continuation_preopt_360.json")
    refined_report["path"] = str(refined_file)
    reports.append(refined_report)

    optimized_360, optimized_360_report = optimize_cyclic_trajectory(
        targets=targets_360,
        initial_beta=refined[BETA_COLS].to_numpy(dtype=float),
        bounds=bounds,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        max_nfev=int(args.max_opt_nfev),
    )
    optimized_360_report = _gate_augmented(optimized_360_report)
    optimized_360_report.update({"path_id": "optimized_360", "source": "refined_continuation", "resolution_points": int(len(optimized_360))})
    optimized_360_file = out_dir / "optimized_360.parquet"
    _write_path_report(
        optimized_360,
        optimized_360_report,
        path_file=optimized_360_file,
        report_file=out_dir / "optimized_360.json",
    )
    optimized_360_report["path"] = str(optimized_360_file)
    reports.append(optimized_360_report)

    final_candidates = [report for report in (refined_report, optimized_360_report) if bool(report.get("centerline_gate_pass", False))]
    if final_candidates:
        selected_360_report = sorted(final_candidates, key=_report_sort_key)[0]
        selected_360 = pd.read_parquet(Path(selected_360_report["path"]))
        selected_360.to_parquet(out_dir / "selected_centerline_360.parquet", index=False, compression="zstd")
        success = True
        reason = "centerline_360_passed"
    else:
        selected_360_report = sorted((refined_report, optimized_360_report), key=_report_sort_key)[0]
        success = False
        reason = "no_passing_360_centerline"
    summary = pd.DataFrame([{key: value for key, value in report.items() if key != "stage_history"} for report in reports])
    summary.to_csv(out_dir / "optimization_summary.csv", index=False)
    write_json(
        out_dir / "optimization_report.json",
        {
            "success": success,
            "reason": reason,
            "selected_72": selected_72_report,
            "selected_360": selected_360_report,
            "reports": reports,
        },
    )
    return summary


def phase_robustness(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "05_robustness"
    paths_dir = out_dir / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)
    selected_input = Path(args.out_dir) / "04_trajectory_optimization" / "selected_centerline_360.parquet"
    optimization_report = _safe_read_json(Path(args.out_dir) / "04_trajectory_optimization" / "optimization_report.json")
    if not selected_input.exists() or not bool(optimization_report.get("success", False)):
        write_json(out_dir / "robustness_report.json", {"robustness_gate_pass": False, "reason": "no_passing_360_centerline"})
        return pd.DataFrame()
    lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    bounds = beta_bounds_rad("current")
    targets = _targets_for_points(args, int(args.final_points))
    targets_path = out_dir / "robustness_targets.parquet"
    targets.to_parquet(targets_path, index=False, compression="zstd")
    task_dir = out_dir / "worker_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    selected_centerline = pd.read_parquet(selected_input).sort_values("angle_idx").reset_index(drop=True)
    v2 = _load_v2_candidates(args)
    angle_zero = v2[v2["angle_idx"].astype(int).eq(0)].sort_values("xyz_residual_mm").reset_index(drop=True)
    start_records: list[dict[str, Any]] = [
        {"start_id": f"v2_{idx:02d}", "start_beta": row[BETA_COLS].to_numpy(dtype=float), "start_source": "v2_angle0"}
        for idx, row in angle_zero.iterrows()
    ]
    canonical_start = selected_centerline.iloc[0][BETA_COLS].to_numpy(dtype=float)
    start_records.append(
        {
            "start_id": "optimized_canonical",
            "start_beta": canonical_start.copy(),
            "start_source": "optimized_centerline",
        }
    )
    rng = np.random.default_rng(int(args.seed))
    random_count = int(args.robustness_random_seeds)
    for idx in range(random_count):
        jitter = rng.normal(0.0, np.deg2rad(0.25), size=6)
        start_records.append(
            {
                "start_id": f"random_{idx:02d}",
                "start_beta": np.clip(canonical_start + jitter, bounds[:, 0], bounds[:, 1]),
                "start_source": "random_jitter",
            }
        )
    if args.preset == "smoke":
        start_records = start_records[:2]

    reports: list[dict[str, Any]] = []
    run_paths: dict[tuple[str, str], pd.DataFrame] = {}
    start_beta_by_id = {str(record["start_id"]): np.asarray(record["start_beta"], dtype=float) for record in start_records}
    pending_tasks: list[Path] = []
    run_outputs: list[tuple[str, str, Path, Path]] = []
    for start_index, start_record in enumerate(start_records):
        for direction in ("forward", "reverse"):
            run_id = f"{start_record['start_id']}_{direction}"
            path_file = paths_dir / f"{run_id}.parquet"
            report_file = paths_dir / f"{run_id}.json"
            run_outputs.append((str(start_record["start_id"]), direction, path_file, report_file))
            if bool(args.skip_existing) and path_file.exists() and report_file.exists():
                continue
            task_path = task_dir / f"{run_id}.json"
            write_json(
                task_path,
                {
                    "kind": "continuation",
                    "robot_config": str(args.robot_config),
                    "targets_path": str(targets_path),
                    "start_beta": np.asarray(start_record["start_beta"], dtype=float),
                    "direction": direction,
                    "method": "warm",
                    "center_lambda": 1.0e-3,
                    "max_ik_nfev": int(args.max_ik_nfev),
                    "run_id": run_id,
                    "start_idx": int(start_index),
                    "start_branch_cluster_id": int(start_index),
                    "path_file": str(path_file),
                    "report_file": str(report_file),
                    "report_metadata": {
                        "start_id": str(start_record["start_id"]),
                        "start_source": str(start_record["start_source"]),
                        "direction": direction,
                    },
                },
            )
            pending_tasks.append(task_path)
    _execute_worker_tasks(pending_tasks, workers=int(args.workers))
    for start_id, direction, path_file, report_file in run_outputs:
        _require_file(path_file, f"robustness path {start_id}/{direction}")
        _require_file(report_file, f"robustness report {start_id}/{direction}")
        path = pd.read_parquet(path_file)
        report = _load_augmented_report(report_file)
        reports.append(report)
        run_paths[(start_id, direction)] = path

    pair_reports: list[dict[str, Any]] = []
    reproducible_pairs: list[dict[str, Any]] = []
    for start_record in start_records:
        start_id = str(start_record["start_id"])
        forward = run_paths.get((start_id, "forward"))
        reverse = run_paths.get((start_id, "reverse"))
        if forward is None or reverse is None:
            continue
        comparison = branch_reproducibility_report(forward, reverse, threshold_deg=1.0)
        f_report = next(report for report in reports if report["run_id"] == f"{start_id}_forward")
        r_report = next(report for report in reports if report["run_id"] == f"{start_id}_reverse")
        record = {
            "start_id": start_id,
            **comparison,
            "forward_centerline_gate_pass": bool(f_report.get("centerline_gate_pass", False)),
            "reverse_centerline_gate_pass": bool(r_report.get("centerline_gate_pass", False)),
            "pair_gate_pass": bool(
                comparison["reproducible"]
                and f_report.get("centerline_gate_pass", False)
                and r_report.get("centerline_gate_pass", False)
            ),
        }
        pair_reports.append(record)
        if record["pair_gate_pass"]:
            reproducible_pairs.append(record)
    pd.DataFrame(pair_reports).to_csv(out_dir / "forward_reverse_reproducibility.csv", index=False)

    passing_reports = [report for report in reports if bool(report.get("centerline_gate_pass", False))]
    pass_ratio = float(len(passing_reports) / max(len(reports), 1))
    if reproducible_pairs:
        pair_by_id = {str(record["start_id"]): record for record in reproducible_pairs}
        eligible = [report for report in passing_reports if str(report["start_id"]) in pair_by_id]
        selected_report = sorted(eligible, key=_report_sort_key)[0]
        selected_path = run_paths[(str(selected_report["start_id"]), str(selected_report["direction"]))]
        selected_pair = pair_by_id[str(selected_report["start_id"])]
    else:
        selected_report = sorted(reports, key=_report_sort_key)[0]
        selected_path = run_paths[(str(selected_report["start_id"]), str(selected_report["direction"]))]
        selected_pair = {"branch_diff_p95_deg": float("inf"), "reproducible": False}

    repeat_path, repeat_report = continuation_lift(
        targets=targets,
        start_beta=start_beta_by_id[str(selected_report["start_id"])],
        bounds=bounds,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        direction=str(selected_report["direction"]),
        method="warm",
        lambda_center=1.0e-3,
        lambda_limit=0.0,
        max_nfev=int(args.max_ik_nfev),
    )
    repeatability = branch_reproducibility_report(selected_path, repeat_path, threshold_deg=1.0e-6)

    cross_start_diffs: list[float] = []
    for i, left in enumerate(passing_reports):
        for right in passing_reports[i + 1 :]:
            comparison = branch_reproducibility_report(
                run_paths[(str(left["start_id"]), str(left["direction"]))],
                run_paths[(str(right["start_id"]), str(right["direction"]))],
                threshold_deg=1.0,
            )
            cross_start_diffs.append(float(comparison["branch_diff_p95_deg"]))
    robustness_gate = _selected_branch_robustness_gate(
        selected_pair=selected_pair,
        repeatability=repeatability,
        selected_report=selected_report,
    )
    if robustness_gate:
        selected_path.to_parquet(out_dir / "selected_centerline_360.parquet", index=False, compression="zstd")
    summary = pd.DataFrame([{key: value for key, value in report.items() if key != "stage_history"} for report in reports])
    summary.to_csv(out_dir / "robustness_runs.csv", index=False)
    write_json(
        out_dir / "robustness_report.json",
        {
            "robustness_gate_pass": robustness_gate,
            "run_pass_ratio": pass_ratio,
            "passing_runs": int(len(passing_reports)),
            "total_runs": int(len(reports)),
            "selected_run": selected_report,
            "selected_forward_reverse": selected_pair,
            "deterministic_repeatability": repeatability,
            "multiple_smooth_branches": bool(cross_start_diffs and max(cross_start_diffs) > 1.0),
            "cross_start_branch_diff_p95_max_deg": float(max(cross_start_diffs)) if cross_start_diffs else 0.0,
            "pair_reports": pair_reports,
        },
    )
    return summary


def phase_tube(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out_dir) / "06_local_tube"
    out_dir.mkdir(parents=True, exist_ok=True)
    centerline_path = Path(args.out_dir) / "05_robustness" / "selected_centerline_360.parquet"
    robustness_path = Path(args.out_dir) / "05_robustness" / "robustness_report.json"
    if not centerline_path.exists() or not robustness_path.exists():
        write_json(
            out_dir / "tube_quality_report.json",
            {
                "tube_gate_pass": False,
                "reason": "no_passing_360_centerline",
                "centerline_path": str(centerline_path),
                "robustness_path": str(robustness_path),
            },
        )
        return pd.DataFrame()
    robustness = _load_robustness_report(robustness_path)
    if not bool(robustness.get("robustness_gate_pass", False)):
        write_json(
            out_dir / "tube_quality_report.json",
            {"tube_gate_pass": False, "reason": "robustness_gate_failed", "robustness": robustness},
        )
        return pd.DataFrame()
    lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    bounds = beta_bounds_rad("current")
    centerline = pd.read_parquet(centerline_path).sort_values("angle_idx").reset_index(drop=True)
    offsets = [-5.0, -2.5, 0.0, 2.5, 5.0]
    if args.preset == "smoke":
        offsets = [-2.5, 0.0, 2.5]
    tube_targets = make_normal_tube_targets(centerline, offsets_mm=offsets)
    tube_targets.to_parquet(out_dir / "tube_targets.parquet", index=False, compression="zstd")
    offset_pairs = [(float(a), float(b)) for a in offsets for b in offsets]
    offset_pairs.sort(key=lambda pair: (math.hypot(pair[0], pair[1]), math.atan2(pair[1], pair[0])))

    center_beta = centerline[BETA_COLS].to_numpy(dtype=float)
    center_xyz = centerline[XYZ_COLS].to_numpy(dtype=float)
    pinv = []
    for beta in center_beta:
        jac = numerical_jacobian_beta(
            beta,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
        )
        pinv.append(weighted_damped_pinv(jac, damping=1.0e-3, weights=np.asarray([4, 4, 2, 2, 1, 1], dtype=float)))
    pinv_path = out_dir / "weighted_pinv.npy"
    np.save(pinv_path, np.asarray(pinv, dtype=float))
    curve_target_dir = out_dir / "curve_targets"
    curve_output_dir = out_dir / "curves"
    task_dir = out_dir / "worker_tasks"
    curve_target_dir.mkdir(parents=True, exist_ok=True)
    curve_output_dir.mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)

    def offset_file_id(pair: tuple[float, float]) -> str:
        return f"n1_{pair[0]:g}_n2_{pair[1]:g}".replace("-", "m").replace(".", "p")

    center_pair = (0.0, 0.0)
    center_targets = tube_targets[
        np.isclose(tube_targets["delta_n1_mm"].astype(float), 0.0)
        & np.isclose(tube_targets["delta_n2_mm"].astype(float), 0.0)
    ].sort_values("angle_idx").reset_index(drop=True)
    center_curve = center_targets.copy()
    for j, col in enumerate(BETA_COLS):
        center_curve[col] = center_beta[:, j]
    center_theta = theta_from_beta_batch(center_beta, theta_sign=theta_sign)
    for j, col in enumerate(THETA_COLS):
        center_curve[col] = center_theta[:, j]
    for j, col in enumerate(XYZ_COLS):
        center_curve[col] = center_xyz[:, j]
    center_curve["xyz_residual_mm"] = np.linalg.norm(
        center_xyz - center_curve[TARGET_XYZ_COLS].to_numpy(dtype=float), axis=1
    ) * 1000.0
    center_curve["tube_success"] = center_curve["xyz_residual_mm"].le(1.5)
    center_curve["parent_offset_id"] = "centerline"
    center_curve_path = curve_output_dir / f"{offset_file_id(center_pair)}.parquet"
    center_curve.to_parquet(center_curve_path, index=False, compression="zstd")
    solved_curve_paths: dict[tuple[float, float], Path] = {center_pair: center_curve_path}

    radius_groups: dict[float, list[tuple[float, float]]] = {}
    for pair in offset_pairs:
        if pair == center_pair:
            continue
        radius_groups.setdefault(round(math.hypot(pair[0], pair[1]), 8), []).append(pair)
    for radius in sorted(radius_groups):
        task_paths: list[Path] = []
        pending_outputs: list[tuple[tuple[float, float], Path]] = []
        for dn1, dn2 in radius_groups[radius]:
            pair = (dn1, dn2)
            curve_targets = tube_targets[
                np.isclose(tube_targets["delta_n1_mm"].astype(float), dn1)
                & np.isclose(tube_targets["delta_n2_mm"].astype(float), dn2)
            ].sort_values("angle_idx").reset_index(drop=True)
            smaller_offsets = [
                old_pair
                for old_pair in solved_curve_paths
                if math.hypot(old_pair[0], old_pair[1]) < radius - 1.0e-12
            ]
            parent_offset = min(
                smaller_offsets,
                key=lambda old_pair: math.hypot(old_pair[0] - dn1, old_pair[1] - dn2),
            ) if smaller_offsets else center_pair
            file_id = offset_file_id(pair)
            target_path = curve_target_dir / f"{file_id}.parquet"
            output_path = curve_output_dir / f"{file_id}.parquet"
            report_path = curve_output_dir / f"{file_id}.json"
            curve_targets.to_parquet(target_path, index=False, compression="zstd")
            pending_outputs.append((pair, output_path))
            if bool(args.skip_existing) and output_path.exists() and report_path.exists():
                continue
            task_path = task_dir / f"{file_id}.json"
            write_json(
                task_path,
                {
                    "kind": "tube_curve",
                    "robot_config": str(args.robot_config),
                    "centerline_path": str(centerline_path),
                    "curve_targets_path": str(target_path),
                    "parent_curve_path": str(solved_curve_paths[parent_offset]),
                    "parent_offset_id": f"n1_{parent_offset[0]:g}_n2_{parent_offset[1]:g}",
                    "pinv_path": str(pinv_path),
                    "max_ik_nfev": int(args.max_ik_nfev),
                    "offset_id": f"n1_{dn1:g}_n2_{dn2:g}",
                    "curve_output": str(output_path),
                    "report_output": str(report_path),
                },
            )
            task_paths.append(task_path)
        _execute_worker_tasks(task_paths, workers=int(args.workers))
        for pair, output_path in pending_outputs:
            _require_file(output_path, f"tube curve {pair}")
            solved_curve_paths[pair] = output_path

    all_rows = [pd.read_parquet(solved_curve_paths[pair]) for pair in offset_pairs]
    tube = pd.concat(all_rows, ignore_index=True)
    tube.to_parquet(out_dir / "tube_attempt.parquet", index=False, compression="zstd")
    residual = tube["xyz_residual_mm"].to_numpy(dtype=float)
    local = local_beta_consistency_report(
        tube,
        radius_mm=10.0,
        max_neighbors=None,
        multi_branch_threshold_deg=3.0,
    )
    success_ratio = float(tube["tube_success"].mean())
    coverage_ratio = float(len(tube) / max(len(tube_targets), 1))
    quality = {
        "rows": int(len(tube)),
        "expected_rows": int(len(tube_targets)),
        "neighbor_radius_mm": 10.0,
        "neighbor_mode": "all_within_radius",
        "multi_branch_threshold_deg": 3.0,
        "target_success_ratio": success_ratio,
        "normal_grid_coverage_ratio": coverage_ratio,
        "residual_p95_mm": float(np.percentile(residual, 95)),
        "residual_max_mm": float(np.max(residual)),
        **local,
    }
    quality["tube_gate_pass"] = bool(
        success_ratio >= 0.99
        and quality["residual_p95_mm"] <= 1.5
        and quality["residual_max_mm"] <= 3.0
        and quality["tube10_beta_rms_p95_deg"] <= 1.0
        and quality["multi_branch_ratio"] == 0.0
        and coverage_ratio >= 0.95
    )
    if quality["tube_gate_pass"]:
        tube.to_parquet(out_dir / "tube_small.parquet", index=False, compression="zstd")
    write_json(out_dir / "tube_quality_report.json", quality)
    pd.DataFrame([quality]).to_csv(out_dir / "tube_quality_summary.csv", index=False)
    return pd.DataFrame([quality])


def _metric_line(label: str, report: Mapping[str, Any], key: str, suffix: str = "") -> str:
    value = report.get(key, "n/a")
    if isinstance(value, float):
        shown = f"{value:.6g}"
    else:
        shown = str(value)
    return f"- {label}: `{shown}{suffix}`"


def phase_summary(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir) / "07_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = _safe_read_json(Path(args.out_dir) / "00_audit" / "audit_report.json")
    continuation = _safe_read_json(Path(args.out_dir) / "01_continuation_ablation" / "selection_report.json")
    enrichment = _safe_read_json(Path(args.out_dir) / "02_candidate_enrichment" / "enrichment_report.json")
    linking = _safe_read_json(Path(args.out_dir) / "03_soft_cyclic_linking" / "selection_report.json")
    optimization = _safe_read_json(Path(args.out_dir) / "04_trajectory_optimization" / "optimization_report.json")
    robustness = _load_robustness_report(Path(args.out_dir) / "05_robustness" / "robustness_report.json")
    tube = _safe_read_json(Path(args.out_dir) / "06_local_tube" / "tube_quality_report.json")
    continuation_report = _gate_augmented(continuation.get("selected_report", {}))
    linking_report = _gate_augmented(linking.get("selected_report", {}))
    selected_72 = _gate_augmented(optimization.get("selected_72", {}))
    optimized_360 = _gate_augmented(optimization.get("selected_360", {}))
    robust_selected_360 = _gate_augmented(robustness.get("selected_run", {}))
    selected_360 = robust_selected_360 if robust_selected_360 else optimized_360
    centerline_exists = bool(optimization.get("success", False))
    robust = bool(robustness.get("robustness_gate_pass", False))
    tube_pass = bool(tube.get("tube_gate_pass", False))
    final = {
        "candidate_id": str(args.candidate_id),
        "continuous_branch_exists": centerline_exists,
        "continuation_branch_gate_pass": bool(continuation_report.get("branch_gate_pass", False)),
        "continuation_centerline_gate_pass": bool(continuation_report.get("centerline_gate_pass", False)),
        "selected_72": selected_72,
        "selected_360": selected_360,
        "optimized_360_intermediate": optimized_360,
        "robustness_selected_360": robust_selected_360,
        "robustness_gate_pass": robust,
        "forward_reverse_branch_diff_p95_deg": float(
            robustness.get("selected_forward_reverse", {}).get("branch_diff_p95_deg", np.inf)
        ),
        "multiple_smooth_branches": bool(robustness.get("multiple_smooth_branches", False)),
        "tube_gate_pass": tube_pass,
        "model_training_allowed_next": bool(centerline_exists and robust and tube_pass),
        "model_training_executed": False,
        "tension_executed": False,
    }
    write_json(out_dir / "final_gate_summary.json", final)

    baseline = audit.get("v2_baseline", {})
    lines = [
        "# E75 真实椭圆连续 Branch Lifting V3 实验总结",
        "",
        f"候选轨迹：`{args.candidate_id}`。本实验仅构造连续 `beta6` branch 与条件式法向 tube；未训练模型，未计算张力。",
        "",
        "## 实验实现与边界",
        "",
        "- 扩展 `scripts/analysis/true_ellipse_atlas_utils.py`：多 seed IK、正反 continuation、null-space seeds、soft cyclic DP、周期稀疏 least-squares、可重复性和全半径邻域一致性诊断。",
        "- 新增 `scripts/analysis/run_true_ellipse_branch_lifting_v3.py`：按 `audit/continuation/enrich/link/optimize/robustness/tube/summary` 分阶段执行，并支持独立 worker 子进程和断点续跑。",
        "- 实验只读取 V2 结果并写入独立 V3 目录，不覆盖 V2。",
        "- 本轮固定 E75，不扩大 Sobol pool、不搜索 E100、不训练模型、不计算张力。",
        "",
        "## 0. V2 基线审计",
        "",
        f"- V2 pointwise rows：`{audit.get('pointwise_rows', 'n/a')}`；angle layers：`{audit.get('angle_layers', 'n/a')}`。",
        f"- 每层 distinct candidates min/p50/p95：`{audit.get('candidate_count_min', 'n/a')}/{audit.get('candidate_count_p50', 'n/a')}/{audit.get('candidate_count_p95', 'n/a')}`。",
        f"- 候选数不超过 2 的层数：`{audit.get('sparse_le2_layers', 'n/a')}`；相邻最小距离大于 1.5 deg 的瓶颈层数：`{audit.get('bottleneck_gt1p5_layers', 'n/a')}`。",
        "- 5 deg hard-link baseline 已从原始 pointwise candidates 重构，并与 V2 threshold sweep 数值一致。",
        "",
        "## 1. 连续 branch 是否存在",
        "",
        f"- 结论：`{'存在' if centerline_exists else '尚未通过 gate'}`。",
        _metric_line("V2 5deg linked delta beta p95", baseline, "delta_beta_rms_p95_deg", " deg"),
        _metric_line("Continuation residual p95", continuation_report, "residual_p95_mm", " mm"),
        _metric_line("Continuation delta beta p95", continuation_report, "delta_beta_p95_deg", " deg"),
        _metric_line("Continuation seam", continuation_report, "seam_beta_rms_deg", " deg"),
        "",
        "## 2. Continuation 与候选增密的作用",
        "",
        f"- Continuation branch gate：`{continuation_report.get('branch_gate_pass', False)}`。",
        f"- Candidate enrichment：`{enrichment.get('success', False)}`。",
        f"- Soft cyclic DP branch gate：`{linking_report.get('branch_gate_pass', False)}`。",
        "- 判定原则：continuation 已显著修复相邻步进，但原始 72 点路径未满足当前 0.75 deg 接缝硬门槛；最终许可由后续优化、360 点 selected-branch robustness 与 tube gate 给出。",
        "",
        "### Candidate density ablation",
        "",
    ]
    for record in enrichment.get("candidate_sets", []):
        lines.append(
            "- `{}`：seed budget `{}`，distinct candidates min/p50/p95/max = `{}/{}/{}/{}`。".format(
                record.get("candidate_set", "unknown"),
                record.get("requested_seed_budget", "n/a"),
                record.get("distinct_candidate_min", "n/a"),
                record.get("distinct_candidate_p50", "n/a"),
                record.get("distinct_candidate_p95", "n/a"),
                record.get("distinct_candidate_max", "n/a"),
            )
        )
    lines += [
        f"- 自适应扩展触发层数：`{enrichment.get('adaptive_trigger_layers', 'n/a')}`。",
        _metric_line("Soft-DP selected delta beta p95", linking_report, "delta_beta_p95_deg", " deg"),
        _metric_line("Soft-DP selected delta2 beta p95", linking_report, "delta2_beta_p95_deg", " deg"),
        _metric_line("Soft-DP selected seam", linking_report, "seam_beta_rms_deg", " deg"),
        "",
        "## 3. 72 点与 360 点 gate",
        "",
        _metric_line("72-point residual p95", selected_72, "residual_p95_mm", " mm"),
        _metric_line("72-point delta beta p95", selected_72, "delta_beta_p95_deg", " deg"),
        _metric_line("72-point delta2 beta p95", selected_72, "delta2_beta_p95_deg", " deg"),
        _metric_line("360-point residual p95", selected_360, "residual_p95_mm", " mm"),
        _metric_line("360-point delta beta p95", selected_360, "delta_beta_p95_deg", " deg"),
        _metric_line("360-point delta2 beta p95", selected_360, "delta2_beta_p95_deg", " deg"),
        _metric_line("360-point seam", selected_360, "seam_beta_rms_deg", " deg"),
        _metric_line("360-point sigma3 p05", selected_360, "sigma3_p05_m", " m"),
        _metric_line("360-point kappa p95", selected_360, "kappa_p95"),
        f"- 72 点最终来源：`{selected_72.get('source', 'n/a')}`；优化中间路径选择 stage：`{optimized_360.get('selected_stage', 'n/a')}`。",
        f"- 最终 tube 中心线来源：`{selected_360.get('run_id', selected_360.get('source', 'n/a'))}`。",
        _metric_line("优化中间路径 seam", optimized_360, "seam_beta_rms_deg", " deg"),
        "- `optimizer_success=false` 仅表示达到本轮 `max_nfev`；路径是否可用仍由独立 tracking、smoothness、conditioning 与正反一致性 gate 决定。",
        "",
        "## 4. 正反向是否收敛到同一 canonical branch",
        "",
        f"- Robustness gate：`{robust}`。",
        _metric_line(
            "Selected forward/reverse branch difference p95",
            robustness.get("selected_forward_reverse", {}),
            "branch_diff_p95_deg",
            " deg",
        ),
        f"- 检测到多个平滑 branch：`{robustness.get('multiple_smooth_branches', False)}`。",
        f"- 不同起点/方向通过率：`{robustness.get('run_pass_ratio', 'n/a')}`。",
        "- Robustness 许可由被选 branch 的正反可复现性、确定性复跑一致性和自身 centerline gate 共同决定；全部 run 通过率仅作为搜索诊断。",
        _metric_line("跨起点 branch difference p95 最大值", robustness, "cross_start_branch_diff_p95_max_deg", " deg"),
        "- 多个平滑 branch 不允许直接混合标签；正式 tube 只使用通过正反一致性 gate 且经字典序 canonical tie-break 选中的单一 branch。",
        "",
        "## 5. Tube 与下一阶段许可",
        "",
        f"- Tube gate：`{tube_pass}`。",
        _metric_line("Tube success ratio", tube, "target_success_ratio"),
        _metric_line("Tube residual p95", tube, "residual_p95_mm", " mm"),
        _metric_line("Tube10 beta RMS p95", tube, "tube10_beta_rms_p95_deg", " deg"),
        _metric_line("Tube multi-branch ratio", tube, "multi_branch_ratio"),
        _metric_line("Tube full-radius neighbor pair count", tube, "neighbor_pair_count"),
        _metric_line("Tube normal-grid coverage", tube, "normal_grid_coverage_ratio"),
        f"- 允许进入下一阶段模型训练：`{final['model_training_allowed_next']}`。",
        "",
        "## 关键产物",
        "",
        "- `00_audit/audit_report.json`",
        "- `01_continuation_ablation/best_continuation.parquet`",
        "- `02_candidate_enrichment/candidates_adaptive.parquet`",
        "- `03_soft_cyclic_linking/best_soft_linked.parquet`",
        "- `04_trajectory_optimization/selected_centerline_360.parquet`",
        "- `05_robustness/selected_centerline_360.parquet`",
        "- `06_local_tube/tube_small.parquet`",
        "- `06_local_tube/tube_quality_report.json`",
        "- `07_summary/final_gate_summary.json`",
        "",
        "## 停止规则",
        "",
        "本次运行遵守 gate：若 360 点 centerline、robustness 或 tube 任一失败，则不会产生可供正式训练使用的 `tube_small.parquet`，也不会启动模型训练。",
        "",
    ]
    summary_path = out_dir / "experiment_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    if args.preset != "smoke":
        docs_path = REPO_ROOT / "docs" / "TrueEllipseBranchLiftingV3实验记录.md"
        docs_path.write_text("\n".join(lines), encoding="utf-8")
    return final


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E75 true-ellipse continuous branch lifting V3.")
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--preset", choices=["smoke", "pilot", "formal"], default="pilot")
    parser.add_argument("--phases", default="all")
    parser.add_argument("--seed-budgets", default="16,32")
    parser.add_argument("--adaptive-seed-budget", type=int, default=64)
    parser.add_argument("--center-lambdas", default="0,0.0001,0.001,0.01")
    parser.add_argument("--coarse-points", type=int, default=72)
    parser.add_argument("--final-points", type=int, default=360)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    parser.add_argument("--max-ik-nfev", type=int, default=100)
    parser.add_argument("--max-opt-nfev", type=int, default=30)
    parser.add_argument("--robustness-random-seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-task", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    apply_preset_defaults(args)
    started = time.perf_counter()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if _phase_selected(args, "audit"):
        phase_audit(args)
    if _phase_selected(args, "continuation"):
        phase_continuation(args)
    if _phase_selected(args, "enrich"):
        phase_enrich(args)
    if _phase_selected(args, "link"):
        phase_link(args)
    if _phase_selected(args, "optimize"):
        phase_optimize(args)
    if _phase_selected(args, "robustness"):
        phase_robustness(args)
    if _phase_selected(args, "tube"):
        phase_tube(args)
    if _phase_selected(args, "summary"):
        phase_summary(args)
    payload = {
        "v2_dir": str(args.v2_dir),
        "out_dir": str(args.out_dir),
        "candidate_id": str(args.candidate_id),
        "preset": str(args.preset),
        "phases": str(args.phases),
        "elapsed_s": float(time.perf_counter() - started),
    }
    write_json(Path(args.out_dir) / "run_report.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    if args.worker_task is not None:
        run_worker_task(Path(args.worker_task))
        return 0
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
