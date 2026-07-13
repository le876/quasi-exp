#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from true_ellipse_family_v5_utils import (  # noqa: E402
    connected_radius_max,
    dataframe_to_markdown,
    file_sha256,
    formal_family_coverage_report,
    parse_float_csv,
    score_family_candidates,
    should_retry_pointwise,
    sobol_family_perturbations,
    stable_fingerprint,
)
import true_ellipse_atlas_utils as atlas  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from true_ellipse_family_v5_utils import family_from_mapping, generate_family_targets  # noqa: E402


DEFAULT_OUT = REPO_ROOT / "runs" / "true_ellipse_family_expansion_v5"
DEFAULT_V2 = REPO_ROOT / "runs" / "true_ellipse_reachability_atlas_v2"
DEFAULT_V3 = REPO_ROOT / "runs" / "true_ellipse_branch_lifting_v3"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml"
ALL_PHASES = ["audit", "search", "pointwise", "branch", "tube", "dataset", "summary"]
BRANCH_SELECTION_STRATEGY_VERSION = 3
BRANCH_REPEATABILITY_STRATEGY_VERSION = 2
FORMAL_TUBE_OFFSETS_MM = (-5.0, -2.5, 0.0, 2.5, 5.0)
FORMAL_RADIUS_ANCHORS_MM = (75.0, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0)
SEARCH_STRATEGY_VERSION = 2
POINTWISE_STRATEGY_VERSION = 2
TUBE_STRATEGY_VERSION = 2
DATASET_STRATEGY_VERSION = 5


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def parse_phases(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        phases = [part.strip() for part in value.split(",") if part.strip()]
    else:
        phases = [str(part).strip() for part in value if str(part).strip()]
    if phases == ["all"]:
        return list(ALL_PHASES)
    unknown = sorted(set(phases) - set(ALL_PHASES))
    if unknown:
        raise ValueError(f"unsupported V5 phases: {unknown}")
    return phases


def validate_formal_tube_offsets(value: str | Iterable[float]) -> list[float]:
    offsets = parse_float_csv(value)
    expected = np.asarray(FORMAL_TUBE_OFFSETS_MM, dtype=float)
    actual = np.asarray(sorted(offsets), dtype=float)
    if len(offsets) != len(expected) or len(set(offsets)) != len(expected) or not np.allclose(
        actual,
        expected,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError(f"formal tube offsets are fixed at {list(FORMAL_TUBE_OFFSETS_MM)} mm")
    return list(FORMAL_TUBE_OFFSETS_MM)


def formal_expansion_protocol_report(args: argparse.Namespace) -> dict[str, Any]:
    def floats(name: str) -> list[float]:
        try:
            return parse_float_csv(getattr(args, name))
        except (AttributeError, TypeError, ValueError):
            return []

    def integers(name: str) -> list[int]:
        try:
            return _parse_int_csv(getattr(args, name))
        except (AttributeError, TypeError, ValueError):
            return []

    offsets = floats("tube_offsets_mm")
    checks = {
        "primary_radius_87p5": np.isclose(float(getattr(args, "primary_radius_mm", np.nan)), 87.5),
        "stretch_radius_100": np.isclose(float(getattr(args, "stretch_radius_mm", np.nan)), 100.0),
        "formal_radius_anchors": bool(
            len(floats("radius_anchors_mm")) == len(FORMAL_RADIUS_ANCHORS_MM)
            and np.allclose(floats("radius_anchors_mm"), FORMAL_RADIUS_ANCHORS_MM, atol=1.0e-12, rtol=0.0)
        ),
        "sobol_samples_2048": int(getattr(args, "family_sobol_samples", -1)) == 2048,
        "pointwise_seed_budgets_16_32_64": integers("pointwise_seed_budgets") == [16, 32, 64],
        "five_pointwise_families": int(getattr(args, "max_pointwise_families", -1)) == 5,
        "five_branch_families": int(getattr(args, "max_branch_families", -1)) == 5,
        "coarse_points_72": int(getattr(args, "coarse_points", -1)) == 72,
        "final_points_360": int(getattr(args, "final_points", -1)) == 360,
        "max_ik_nfev_200": int(getattr(args, "max_ik_nfev", -1)) == 200,
        "max_opt_nfev_40": int(getattr(args, "max_opt_nfev", -1)) == 40,
        "formal_seed_20260713": int(getattr(args, "seed", -1)) == 20260713,
        "formal_tube_offsets": bool(
            len(offsets) == len(FORMAL_TUBE_OFFSETS_MM)
            and len(set(offsets)) == len(FORMAL_TUBE_OFFSETS_MM)
            and np.allclose(sorted(offsets), FORMAL_TUBE_OFFSETS_MM, atol=1.0e-12, rtol=0.0)
        ),
    }
    normalized_checks = {name: bool(value) for name, value in checks.items()}
    protocol = {
        "primary_radius_mm": float(getattr(args, "primary_radius_mm", np.nan)),
        "stretch_radius_mm": float(getattr(args, "stretch_radius_mm", np.nan)),
        "radius_anchors_mm": floats("radius_anchors_mm"),
        "family_sobol_samples": int(getattr(args, "family_sobol_samples", -1)),
        "pointwise_seed_budgets": integers("pointwise_seed_budgets"),
        "max_pointwise_families": int(getattr(args, "max_pointwise_families", -1)),
        "max_branch_families": int(getattr(args, "max_branch_families", -1)),
        "coarse_points": int(getattr(args, "coarse_points", -1)),
        "final_points": int(getattr(args, "final_points", -1)),
        "max_ik_nfev": int(getattr(args, "max_ik_nfev", -1)),
        "max_opt_nfev": int(getattr(args, "max_opt_nfev", -1)),
        "seed": int(getattr(args, "seed", -1)),
        "tube_offsets_mm": offsets,
    }
    return {
        "formal_expansion_protocol_gate_pass": bool(all(normalized_checks.values())),
        "checks": normalized_checks,
        "protocol": protocol,
        "protocol_fingerprint": stable_fingerprint(protocol),
    }


def phase_cache_is_compatible(
    report: Mapping[str, Any],
    *,
    strategy_version: int,
    task_fingerprint: str,
) -> bool:
    return bool(
        int(report.get("strategy_version", -1)) == int(strategy_version)
        and str(report.get("task_fingerprint", "")) == str(task_fingerprint)
    )


def _phase_manifest(
    directory: Path,
    *,
    strategy_version: int,
    task_fingerprint: str,
    skip_existing: bool,
) -> bool:
    path = Path(directory) / "phase_manifest.json"
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    reusable = bool(
        skip_existing
        and phase_cache_is_compatible(
            cached,
            strategy_version=int(strategy_version),
            task_fingerprint=str(task_fingerprint),
        )
    )
    write_json(
        path,
        {
            "strategy_version": int(strategy_version),
            "task_fingerprint": str(task_fingerprint),
        },
    )
    return reusable


def summarize_pointwise_candidates(candidates: pd.DataFrame, *, target_count: int) -> dict[str, Any]:
    required = {"angle_idx", "xyz_residual_mm"}
    missing_columns = sorted(required - set(candidates.columns))
    if missing_columns:
        raise ValueError(f"pointwise candidates missing columns: {missing_columns}")
    best = candidates.groupby("angle_idx", sort=True)["xyz_residual_mm"].min()
    expected = set(range(int(target_count)))
    covered = set(int(value) for value in best.index)
    failed = sorted((expected - covered) | {int(idx) for idx, value in best.items() if float(value) > 2.0})
    residual = best.to_numpy(dtype=float)
    residual_p95 = float(np.percentile(residual, 95)) if len(residual) else float("inf")
    residual_max = float(np.max(residual)) if len(residual) else float("inf")
    success_count = int(sum(float(value) <= 2.0 for value in best))
    gate = bool(len(covered) == int(target_count) and not failed)
    retry = should_retry_pointwise(
        target_count=int(target_count),
        failed_angle_indices=failed,
        residual_max_mm=residual_max,
    )
    return {
        "target_count": int(target_count),
        "covered_target_count": int(len(covered)),
        "success_target_count": success_count,
        "success_ratio": float(success_count / max(int(target_count), 1)),
        "failed_angle_indices": failed,
        "residual_p95_mm": residual_p95,
        "residual_max_mm": residual_max,
        "pointwise_gate_pass": gate,
        "retry_allowed": bool(retry),
    }


def _deduplicate_beta_rows(beta: np.ndarray, *, decimals: int = 10) -> np.ndarray:
    values = np.asarray(beta, dtype=float).reshape(-1, 6)
    if len(values) <= 1:
        return values
    _unique, indices = np.unique(np.round(values, decimals=int(decimals)), axis=0, return_index=True)
    return values[np.sort(indices)]


def solve_pointwise_targets(
    targets: pd.DataFrame,
    *,
    pool: pd.DataFrame,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    seed_budget: int,
    max_nfev: int,
    workers: int,
    parent_path: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required_pool = {*atlas.XYZ_COLS, *atlas.BETA_COLS}
    missing_pool = sorted(required_pool - set(pool.columns))
    if missing_pool:
        raise ValueError(f"pointwise pool missing columns: {missing_pool}")
    required_targets = {"angle_idx", *atlas.TARGET_XYZ_COLS}
    missing_targets = sorted(required_targets - set(targets.columns))
    if missing_targets:
        raise ValueError(f"pointwise targets missing columns: {missing_targets}")
    pool_xyz = pool[atlas.XYZ_COLS].to_numpy(dtype=float)
    tree = cKDTree(pool_xyz)
    k = min(max(1, int(seed_budget)), len(pool))
    parent_by_angle: dict[int, np.ndarray] = {}
    if parent_path is not None and not parent_path.empty:
        parent_by_angle = {
            int(row["angle_idx"]): row[atlas.BETA_COLS].to_numpy(dtype=float)
            for _idx, row in parent_path.iterrows()
        }
    bounds = atlas.beta_bounds_rad("current")

    def solve_one(row_dict: dict[str, Any]) -> list[dict[str, Any]]:
        target_row = pd.Series(row_dict)
        target_xyz = target_row[atlas.TARGET_XYZ_COLS].to_numpy(dtype=float)
        _distance, nearest = tree.query(target_xyz, k=k)
        indices = np.atleast_1d(nearest).astype(int)
        seeds = pool.iloc[indices][atlas.BETA_COLS].to_numpy(dtype=float)
        angle_idx = int(target_row["angle_idx"])
        if angle_idx in parent_by_angle:
            seeds = np.vstack([parent_by_angle[angle_idx], seeds])
        seeds = _deduplicate_beta_rows(seeds)
        records: list[dict[str, Any]] = []
        for seed_rank, seed_beta in enumerate(seeds):
            solution = atlas.solve_beta_ik_many(
                target_xyz,
                init_betas=np.asarray(seed_beta, dtype=float).reshape(1, 6),
                bounds=bounds,
                lengths_m=np.asarray(lengths_m, dtype=float),
                p_end_local_m=np.asarray(p_end_local_m, dtype=float),
                theta_sign=float(theta_sign),
                max_nfev=int(max_nfev),
                lambda_limit=0.0,
                lambda_center=0.0,
            )[0]
            record = dict(row_dict)
            record.update(
                {
                    "seed_rank": int(seed_rank),
                    "inverse_nfev": int(solution.nfev),
                    "xyz_residual_mm": float(solution.residual_mm),
                    "ik_success": bool(solution.residual_mm <= 2.0),
                }
            )
            for idx, column in enumerate(atlas.BETA_COLS):
                record[column] = float(solution.beta_rad[idx])
            for idx, column in enumerate(atlas.XYZ_COLS):
                record[column] = float(solution.xyz_m[idx])
            records.append(record)
            if solution.residual_mm <= 2.0:
                break
        return records

    target_records = targets.to_dict(orient="records")
    if int(workers) <= 1:
        nested = [solve_one(record) for record in target_records]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            nested = list(executor.map(solve_one, target_records))
    candidates = pd.DataFrame([record for group in nested for record in group])
    best_indices = candidates.groupby("angle_idx", sort=True)["xyz_residual_mm"].idxmin()
    best = candidates.loc[best_indices].sort_values("angle_idx").reset_index(drop=True)
    report = summarize_pointwise_candidates(candidates, target_count=len(targets))
    report.update({"seed_budget": int(seed_budget), "candidate_rows": int(len(candidates))})
    return candidates, best, report


def _solve_prepared_pointwise_item(
    item: Mapping[str, Any],
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    max_nfev: int,
) -> list[dict[str, Any]]:
    target_row = dict(item["target_row"])
    target_xyz = np.asarray([target_row[column] for column in atlas.TARGET_XYZ_COLS], dtype=float)
    seeds = _deduplicate_beta_rows(np.asarray(item["seed_betas"], dtype=float))
    bounds = atlas.beta_bounds_rad("current")
    records: list[dict[str, Any]] = []
    for seed_rank, seed_beta in enumerate(seeds):
        solution = atlas.solve_beta_ik_many(
            target_xyz,
            init_betas=np.asarray(seed_beta, dtype=float).reshape(1, 6),
            bounds=bounds,
            lengths_m=np.asarray(lengths_m, dtype=float),
            p_end_local_m=np.asarray(p_end_local_m, dtype=float),
            theta_sign=float(theta_sign),
            max_nfev=int(max_nfev),
            lambda_limit=0.0,
            lambda_center=0.0,
        )[0]
        record = dict(target_row)
        record.update(
            {
                "seed_rank": int(seed_rank),
                "inverse_nfev": int(solution.nfev),
                "xyz_residual_mm": float(solution.residual_mm),
                "ik_success": bool(solution.residual_mm <= 2.0),
            }
        )
        for idx, column in enumerate(atlas.BETA_COLS):
            record[column] = float(solution.beta_rad[idx])
        for idx, column in enumerate(atlas.XYZ_COLS):
            record[column] = float(solution.xyz_m[idx])
        records.append(record)
        if solution.residual_mm <= 2.0:
            break
    return records


def run_worker_task(task_path: Path) -> None:
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    kind = str(task.get("kind", ""))
    worker_args = argparse.Namespace(robot_config=Path(task["robot_config"]))
    lengths_m, p_end_local_m, theta_sign = _load_robot(worker_args)
    if kind == "pointwise_chunk":
        nested = [
            _solve_prepared_pointwise_item(
                item,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
                max_nfev=int(task["max_nfev"]),
            )
            for item in task["items"]
        ]
        candidates = pd.DataFrame([record for group in nested for record in group])
        output_path = Path(task["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_parquet(output_path, index=False, compression="zstd")
        write_json(
            Path(task["report_path"]),
            {
                "task_id": str(task["task_id"]),
                "task_fingerprint": str(task.get("task_fingerprint", "")),
                "target_count": int(len(task["items"])),
                "candidate_rows": int(len(candidates)),
                "output_path": str(output_path),
            },
        )
        return
    if kind == "tube_curve":
        curve = solve_tube_curve(
            pd.read_parquet(task["curve_targets_path"]),
            centerline=pd.read_parquet(task["centerline_path"]),
            parent_curve=pd.read_parquet(task["parent_curve_path"]),
            weighted_pinv=np.load(task["pinv_path"]),
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
            max_nfev=int(task["max_nfev"]),
            parent_offset_id=str(task["parent_offset_id"]),
        )
        output_path = Path(task["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        curve.to_parquet(output_path, index=False, compression="zstd")
        write_json(
            Path(task["report_path"]),
            {
                "task_id": str(task["task_id"]),
                "task_fingerprint": str(task.get("task_fingerprint", "")),
                "offset_id": str(task["offset_id"]),
                "rows": int(len(curve)),
                "success_ratio": float(curve["tube_success"].mean()),
                "residual_p95_mm": float(np.percentile(curve["xyz_residual_mm"], 95)),
                "residual_max_mm": float(curve["xyz_residual_mm"].max()),
            },
        )
        return
    if kind == "branch_continuation":
        targets = pd.read_parquet(task["targets_path"])
        lifted, raw = atlas.continuation_lift(
            targets=targets,
            start_beta=np.asarray(task["start_beta"], dtype=float),
            bounds=atlas.beta_bounds_rad("current"),
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
            direction=str(task["direction"]),
            method="warm",
            lambda_center=1.0e-3,
            lambda_limit=0.0,
            max_nfev=int(task["max_nfev"]),
        )
        report = _augmented_centerline_report(
            raw,
            source=str(task["source"]),
            direction=str(task["direction"]),
            resolution_points=int(len(lifted)),
        )
        output_path = Path(task["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lifted.to_parquet(output_path, index=False, compression="zstd")
        write_json(Path(task["report_path"]), report)
        return
    raise ValueError(f"unsupported V5 worker task kind: {kind}")


def _execute_worker_tasks(task_paths: Iterable[Path], *, workers: int) -> None:
    paths = [Path(path) for path in task_paths]
    if not paths:
        return

    def execute(path: Path) -> None:
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-task", str(path)],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"V5 worker failed for {path} (exit {result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        list(executor.map(execute, paths))


def solve_pointwise_targets_subprocess(
    targets: pd.DataFrame,
    *,
    pool: pd.DataFrame,
    robot_config: str | Path,
    seed_budget: int,
    max_nfev: int,
    workers: int,
    work_dir: Path,
    skip_existing: bool,
    parent_path: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required_pool = {*atlas.XYZ_COLS, *atlas.BETA_COLS}
    required_targets = {"angle_idx", *atlas.TARGET_XYZ_COLS}
    missing_pool = sorted(required_pool - set(pool.columns))
    missing_targets = sorted(required_targets - set(targets.columns))
    if missing_pool or missing_targets:
        raise ValueError(f"subprocess pointwise inputs missing columns: pool={missing_pool}, targets={missing_targets}")
    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pool_xyz = pool[atlas.XYZ_COLS].to_numpy(dtype=float)
    tree = cKDTree(pool_xyz)
    k = min(max(1, int(seed_budget)), len(pool))
    parent_by_angle: dict[int, np.ndarray] = {}
    if parent_path is not None and not parent_path.empty:
        parent_by_angle = {
            int(row["angle_idx"]): row[atlas.BETA_COLS].to_numpy(dtype=float)
            for _idx, row in parent_path.iterrows()
        }
    items: list[dict[str, Any]] = []
    for target_row in targets.to_dict(orient="records"):
        target_xyz = np.asarray([target_row[column] for column in atlas.TARGET_XYZ_COLS], dtype=float)
        _distance, nearest = tree.query(target_xyz, k=k)
        indices = np.atleast_1d(nearest).astype(int)
        seeds = pool.iloc[indices][atlas.BETA_COLS].to_numpy(dtype=float)
        angle_idx = int(target_row["angle_idx"])
        if angle_idx in parent_by_angle:
            seeds = np.vstack([parent_by_angle[angle_idx], seeds])
        items.append({"target_row": target_row, "seed_betas": _deduplicate_beta_rows(seeds).tolist()})
    task_count = min(len(items), max(1, int(workers) * 4))
    chunks = [list(chunk) for chunk in np.array_split(np.asarray(items, dtype=object), task_count) if len(chunk)]
    task_paths: list[Path] = []
    output_paths: list[Path] = []
    for task_index, chunk in enumerate(chunks):
        task_id = f"chunk_{task_index:03d}"
        task_path = directory / f"{task_id}.json"
        output_path = directory / f"{task_id}.parquet"
        report_path = directory / f"{task_id}_report.json"
        output_paths.append(output_path)
        task_fingerprint = stable_fingerprint(
            {
                "kind": "pointwise_chunk",
                "strategy_version": POINTWISE_STRATEGY_VERSION,
                "robot_config": _file_hash_or_missing(robot_config),
                "max_nfev": int(max_nfev),
                "items": chunk,
            }
        )
        if bool(skip_existing) and output_path.exists() and report_path.exists():
            cached = json.loads(report_path.read_text(encoding="utf-8"))
            if str(cached.get("task_fingerprint", "")) == task_fingerprint:
                continue
        write_json(
            task_path,
            {
                "kind": "pointwise_chunk",
                "task_id": task_id,
                "robot_config": str(robot_config),
                "max_nfev": int(max_nfev),
                "items": chunk,
                "task_fingerprint": task_fingerprint,
                "output_path": str(output_path),
                "report_path": str(report_path),
            },
        )
        task_paths.append(task_path)
    _execute_worker_tasks(task_paths, workers=int(workers))
    missing_outputs = [str(path) for path in output_paths if not path.exists()]
    if missing_outputs:
        raise RuntimeError(f"pointwise subprocess workers missed outputs: {missing_outputs}")
    candidates = pd.concat([pd.read_parquet(path) for path in output_paths], ignore_index=True)
    best = _best_rows(candidates)
    report = summarize_pointwise_candidates(candidates, target_count=len(targets))
    report.update(
        {
            "seed_budget": int(seed_budget),
            "candidate_rows": int(len(candidates)),
            "execution_backend": "subprocess",
            "worker_count": int(workers),
            "worker_task_count": int(len(chunks)),
        }
    )
    return candidates, best, report


def family_seeds_from_sources(
    v3_selected: pd.DataFrame,
    v2_candidates: pd.DataFrame,
    *,
    stretch_radius_mm: float,
    max_stretch_centers: int = 2,
) -> pd.DataFrame:
    geometry_cols = ["center_x_m", "center_y_m", "center_z_m", "phase_y_rad", "phase_z_rad"]
    missing_v3 = sorted(set(geometry_cols) - set(v3_selected.columns))
    missing_v2 = sorted({*geometry_cols, "amp_xy_mm", "candidate_id"} - set(v2_candidates.columns))
    if missing_v3 or missing_v2:
        raise ValueError(f"family seed sources missing columns: v3={missing_v3}, v2={missing_v2}")
    rows: list[dict[str, Any]] = []
    if not v3_selected.empty:
        first = v3_selected.iloc[0]
        rows.append(
            {
                "family_id": "v3_selected",
                "source_kind": "v3_selected",
                "source_candidate_id": str(first.get("candidate_id", "v3_selected")),
                **{column: float(first[column]) for column in geometry_cols},
            }
        )
    stretch = v2_candidates[
        np.isclose(v2_candidates["amp_xy_mm"].to_numpy(dtype=float), float(stretch_radius_mm), atol=1.0e-8)
    ]
    center_key = stretch[["center_x_m", "center_y_m", "center_z_m"]].round(10).astype(str).agg("|".join, axis=1)
    stretch = stretch.loc[~center_key.duplicated()].head(max(0, int(max_stretch_centers)))
    for _idx, row in stretch.iterrows():
        rows.append(
            {
                "family_id": str(row["candidate_id"]),
                "source_kind": "v2_stretch",
                "source_candidate_id": str(row["candidate_id"]),
                **{column: float(row[column]) for column in geometry_cols},
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    geometry_key = result[geometry_cols].round(10).astype(str).agg("|".join, axis=1)
    return result.loc[~geometry_key.duplicated()].reset_index(drop=True)


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "reachability_pool": Path(args.v2_dir) / "01_reachability_pool" / "reachability_pool_merged.parquet",
        "v2_candidates": Path(args.v2_dir) / "02_ellipse_family_search" / "top_candidates_by_radius.csv",
        "v3_centerline": Path(args.v3_dir) / "05_robustness" / "selected_centerline_360.parquet",
        "v3_tube": Path(args.v3_dir) / "06_local_tube" / "tube_small.parquet",
    }


def _input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def _file_hash_or_missing(path: str | Path) -> str:
    source = Path(path)
    return file_sha256(source) if source.exists() else "missing"


def _arg_path(args: argparse.Namespace, name: str, default: Path) -> Path:
    return Path(getattr(args, name, default))


def search_task_fingerprint(args: argparse.Namespace) -> str:
    paths = _source_paths(args)
    payload = {
        "phase": "search",
        "strategy_version": SEARCH_STRATEGY_VERSION,
        "protocol": formal_expansion_protocol_report(args)["protocol"],
        "inputs": _input_hashes(paths),
    }
    return stable_fingerprint(payload)


def ensure_search_report(args: argparse.Namespace) -> dict[str, Any]:
    report_path = Path(args.out_dir) / "01_family_search" / "search_report.json"
    if all(path.exists() for path in _source_paths(args).values()):
        expected = search_task_fingerprint(args)
        if report_path.exists():
            cached = json.loads(report_path.read_text(encoding="utf-8"))
            if phase_cache_is_compatible(
                cached,
                strategy_version=SEARCH_STRATEGY_VERSION,
                task_fingerprint=expected,
            ):
                return cached
    return phase_search(args)


def phase_audit(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "00_audit"
    paths = _source_paths(args)
    checks = {name: path.exists() for name, path in paths.items()}
    report: dict[str, Any] = {
        "sources": {name: str(path) for name, path in paths.items()},
        "checks": checks,
        "audit_gate_pass": bool(all(checks.values())),
    }
    if checks["v3_centerline"]:
        centerline = pd.read_parquet(paths["v3_centerline"])
        report["v3_centerline_rows"] = int(len(centerline))
        report["v3_centerline_candidates"] = sorted(centerline.get("candidate_id", pd.Series(dtype=str)).astype(str).unique().tolist())
    if checks["v3_tube"]:
        tube = pd.read_parquet(paths["v3_tube"])
        report["v3_tube_rows"] = int(len(tube))
    write_json(out / "audit_report.json", report)
    if not report["audit_gate_pass"]:
        missing = [name for name, passed in checks.items() if not passed]
        raise FileNotFoundError(f"V5 source audit failed; missing {missing}")
    return report


def _search_candidate_table(seeds: pd.DataFrame, *, samples_per_seed: int, seed: int) -> pd.DataFrame:
    perturbed = sobol_family_perturbations(
        seeds,
        samples_per_seed=int(samples_per_seed),
        seed=int(seed),
    )
    originals = seeds.copy()
    originals["candidate_id"] = originals["family_id"].astype(str)
    originals["source_family_id"] = originals["family_id"].astype(str)
    return pd.concat([originals, perturbed], ignore_index=True, sort=False)


def phase_search(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "01_family_search"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "search_report.json"
    paths = _source_paths(args)
    if not all(path.exists() for path in paths.values()):
        phase_audit(args)
    task_fingerprint = search_task_fingerprint(args)
    if bool(getattr(args, "skip_existing", False)) and report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if phase_cache_is_compatible(
            cached,
            strategy_version=SEARCH_STRATEGY_VERSION,
            task_fingerprint=task_fingerprint,
        ):
            return cached
    centerline = pd.read_parquet(paths["v3_centerline"])
    v2_candidates = pd.read_csv(paths["v2_candidates"])
    seeds = family_seeds_from_sources(centerline, v2_candidates, stretch_radius_mm=float(args.stretch_radius_mm))
    if seeds.empty:
        raise RuntimeError("V5 family search found no source families")
    seeds.to_csv(out / "family_seeds.csv", index=False)
    candidates = _search_candidate_table(
        seeds,
        samples_per_seed=int(args.family_sobol_samples),
        seed=int(args.seed),
    )
    candidates.to_parquet(out / "family_candidates.parquet", index=False, compression="zstd")
    pool = pd.read_parquet(paths["reachability_pool"], columns=["x_m", "y_m", "z_m"])
    pool_xyz = pool[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    anchors = parse_float_csv(args.radius_anchors_mm)
    primary_anchors = [radius for radius in anchors if radius <= float(args.primary_radius_mm) + 1.0e-9]
    stretch_anchors = [radius for radius in anchors if radius <= float(args.stretch_radius_mm) + 1.0e-9]
    primary_ranking, primary_detail = score_family_candidates(
        candidates,
        pool_xyz=pool_xyz,
        radii_mm=primary_anchors,
        n_points=int(args.coarse_points),
    )
    stretch_ranking, stretch_detail = score_family_candidates(
        candidates,
        pool_xyz=pool_xyz,
        radii_mm=stretch_anchors,
        n_points=int(args.coarse_points),
    )
    primary_ranking.to_csv(out / "primary_family_ranking.csv", index=False)
    primary_detail.to_parquet(out / "primary_support_by_radius.parquet", index=False, compression="zstd")
    stretch_ranking.to_csv(out / "stretch_family_ranking.csv", index=False)
    stretch_detail.to_parquet(out / "stretch_support_by_radius.parquet", index=False, compression="zstd")
    report = {
        "strategy_version": SEARCH_STRATEGY_VERSION,
        "task_fingerprint": task_fingerprint,
        **formal_expansion_protocol_report(args),
        "source_seed_count": int(len(seeds)),
        "candidate_count": int(len(candidates)),
        "primary_anchors_mm": primary_anchors,
        "stretch_anchors_mm": stretch_anchors,
        "primary_selected_candidate_id": str(primary_ranking.iloc[0]["candidate_id"]),
        "stretch_selected_candidate_id": str(stretch_ranking.iloc[0]["candidate_id"]),
        "primary_selected_metrics": primary_ranking.iloc[0].to_dict(),
        "stretch_selected_metrics": stretch_ranking.iloc[0].to_dict(),
        "primary_ranking_path": str(out / "primary_family_ranking.csv"),
        "stretch_ranking_path": str(out / "stretch_family_ranking.csv"),
    }
    write_json(report_path, report)
    return report


def select_pointwise_families(
    primary_ranking: pd.DataFrame,
    stretch_ranking: pd.DataFrame,
    seeds: pd.DataFrame,
    *,
    max_families: int,
) -> pd.DataFrame:
    limit = max(1, int(max_families))
    frames: list[pd.DataFrame] = []
    baseline = seeds[seeds["family_id"].astype(str).eq("v3_selected")].head(1).copy()
    if not baseline.empty:
        baseline["candidate_id"] = baseline["family_id"].astype(str)
        frames.append(baseline)
    remaining = limit - sum(len(frame) for frame in frames)
    primary_slots = int(math.ceil(remaining / 2.0))
    stretch_slots = remaining - primary_slots
    frames.append(primary_ranking.head(primary_slots).copy())
    frames.append(stretch_ranking.head(stretch_slots).copy())
    selected = pd.concat(frames, ignore_index=True, sort=False)
    selected = selected.drop_duplicates("candidate_id", keep="first")
    if len(selected) < limit:
        fallback = pd.concat([primary_ranking, stretch_ranking], ignore_index=True, sort=False)
        selected = pd.concat([selected, fallback], ignore_index=True, sort=False).drop_duplicates("candidate_id", keep="first")
    return selected.head(limit).reset_index(drop=True)


def _load_robot(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    config = load_config(str(args.robot_config))
    inputs = load_robot_inputs(config)
    theta_sign = float(config.get("kinematics", {}).get("theta_sign", -1.0))
    return inputs.lengths_m, inputs.p_end_local_m, theta_sign


def _parse_int_csv(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def _radius_slug(radius_mm: float) -> str:
    return f"r{float(radius_mm):06.2f}".replace(".", "p")


def _resample_parent_path(parent: pd.DataFrame, *, target_count: int) -> pd.DataFrame:
    ordered = parent.sort_values("angle_idx").reset_index(drop=True)
    if len(ordered) == int(target_count):
        return ordered
    positions = np.round(np.arange(int(target_count)) * len(ordered) / int(target_count)).astype(int) % len(ordered)
    sampled = ordered.iloc[positions].copy().reset_index(drop=True)
    sampled["angle_idx"] = np.arange(int(target_count), dtype=np.int64)
    return sampled


def _best_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    indices = candidates.groupby("angle_idx", sort=True)["xyz_residual_mm"].idxmin()
    return candidates.loc[indices].sort_values("angle_idx").reset_index(drop=True)


def periodic_resample_beta(path: pd.DataFrame, *, target_count: int) -> np.ndarray:
    """Linearly resample one closed beta branch without dropping the seam interval."""
    ordered = path.sort_values("angle_idx").reset_index(drop=True)
    if len(ordered) < 2:
        raise ValueError("periodic beta resampling requires at least two source rows")
    count = int(target_count)
    if count < 2:
        raise ValueError("periodic beta resampling requires at least two target rows")
    beta = ordered[atlas.BETA_COLS].to_numpy(dtype=float)
    source_phase = np.arange(len(beta) + 1, dtype=float) / float(len(beta))
    wrapped = np.vstack([beta, beta[0]])
    target_phase = np.arange(count, dtype=float) / float(count)
    return np.column_stack(
        [np.interp(target_phase, source_phase, wrapped[:, column]) for column in range(beta.shape[1])]
    )


def evaluate_branch_robustness(
    *,
    centerline_report: Mapping[str, Any],
    forward_reverse: Mapping[str, Any],
    repeatability: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the unchanged V3 centerline and reproducibility hard gates."""
    report = dict(centerline_report)
    report.update(atlas.evaluate_centerline_gates(report))
    forward_reverse_pass = bool(
        forward_reverse.get("reproducible", False)
        and float(forward_reverse.get("branch_diff_p95_deg", np.inf)) <= 1.0
    )
    repeatability_pass = bool(repeatability.get("reproducible", False))
    report.update(
        {
            "forward_reverse_gate_pass": forward_reverse_pass,
            "deterministic_repeatability_gate_pass": repeatability_pass,
            "branch_robustness_gate_pass": bool(
                report["centerline_gate_pass"] and forward_reverse_pass and repeatability_pass
            ),
        }
    )
    return report


def tube_gate_pass(metrics: Mapping[str, Any]) -> bool:
    """Return the unchanged V3 5x5 normal-tube acceptance decision."""
    return bool(
        int(metrics.get("normal_grid_size", 0)) == 25
        and float(metrics.get("target_success_ratio", -np.inf)) >= 0.99
        and float(metrics.get("normal_grid_coverage_ratio", -np.inf)) >= 0.95
        and float(metrics.get("residual_p95_mm", np.inf)) <= 1.5
        and float(metrics.get("residual_max_mm", np.inf)) <= 3.0
        and float(metrics.get("tube10_beta_rms_p95_deg", np.inf)) <= 1.0
        and float(metrics.get("multi_branch_ratio", np.inf)) == 0.0
    )


def select_branch_families(
    pointwise_summary: pd.DataFrame,
    *,
    primary_radius_mm: float,
    stretch_radius_mm: float,
    max_families: int,
    primary_preference: Iterable[str] = (),
    stretch_preference: Iterable[str] = (),
) -> list[str]:
    """Reserve capacity for both the primary-radius and stretch-radius contenders."""
    passing = pointwise_summary[pointwise_summary["pointwise_gate_pass"].fillna(False).astype(bool)].copy()
    if passing.empty:
        return []

    def ranked_at(radius_mm: float, preference: Iterable[str]) -> list[str]:
        mask = np.isclose(passing["radius_mm"].to_numpy(dtype=float), float(radius_mm), atol=1.0e-8)
        part = passing.loc[mask].sort_values(["residual_p95_mm", "candidate_id"], kind="stable")
        available = part["candidate_id"].astype(str).tolist()
        preferred = [str(candidate_id) for candidate_id in preference if str(candidate_id) in available]
        return [*preferred, *(candidate_id for candidate_id in available if candidate_id not in preferred)]

    primary = ranked_at(float(primary_radius_mm), primary_preference)
    stretch = ranked_at(float(stretch_radius_mm), stretch_preference)
    ordered: list[str] = []

    def add(candidate_id: str) -> None:
        if candidate_id not in ordered:
            ordered.append(candidate_id)

    if stretch:
        add(stretch[0])
    if primary:
        add(primary[0])
    for rank in range(1, max(len(primary), len(stretch))):
        if rank < len(stretch):
            add(stretch[rank])
        if rank < len(primary):
            add(primary[rank])
    fallback = (
        passing.groupby("candidate_id", as_index=False)
        .agg(max_radius_mm=("radius_mm", "max"), best_residual_p95_mm=("residual_p95_mm", "min"))
        .sort_values(["max_radius_mm", "best_residual_p95_mm", "candidate_id"], ascending=[False, True, True])
    )
    for candidate_id in fallback["candidate_id"].astype(str):
        add(candidate_id)
    return ordered[: max(0, int(max_families))]


def family_covers_both_goals(
    passing_radii_mm: Iterable[float],
    *,
    primary_radius_mm: float,
    stretch_radius_mm: float,
) -> bool:
    values = np.asarray(parse_float_csv(passing_radii_mm), dtype=float)
    return bool(
        np.any(np.isclose(values, float(primary_radius_mm), atol=1.0e-8))
        and np.any(np.isclose(values, float(stretch_radius_mm), atol=1.0e-8))
    )


def solve_tube_curve(
    curve_targets: pd.DataFrame,
    *,
    centerline: pd.DataFrame,
    parent_curve: pd.DataFrame,
    weighted_pinv: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    max_nfev: int,
    parent_offset_id: str,
) -> pd.DataFrame:
    """Lift one fixed normal-offset curve while staying on the selected branch."""
    targets = curve_targets.sort_values("angle_idx").reset_index(drop=True)
    center = centerline.sort_values("angle_idx").reset_index(drop=True)
    parent = parent_curve.sort_values("angle_idx").reset_index(drop=True)
    if len(targets) != len(center) or len(parent) != len(center):
        raise ValueError("tube curve, centerline, and parent curve must have identical resolution")
    pinv = np.asarray(weighted_pinv, dtype=float).reshape(len(center), 6, 3)
    center_beta = center[atlas.BETA_COLS].to_numpy(dtype=float)
    center_xyz = center[atlas.XYZ_COLS].to_numpy(dtype=float)
    bounds = atlas.beta_bounds_rad("current")
    rows: list[dict[str, Any]] = []
    previous_beta: np.ndarray | None = None
    for position, target_row in targets.iterrows():
        target_xyz = target_row[atlas.TARGET_XYZ_COLS].to_numpy(dtype=float)
        prediction = np.clip(
            center_beta[position] + pinv[position] @ (target_xyz - center_xyz[position]),
            bounds[:, 0],
            bounds[:, 1],
        )
        parent_beta = parent.iloc[position][atlas.BETA_COLS].to_numpy(dtype=float)
        seeds = [prediction, parent_beta, center_beta[position]]
        if previous_beta is not None:
            seeds.insert(0, previous_beta)
        seeds_array = _deduplicate_beta_rows(np.vstack(seeds))
        solutions = atlas.solve_beta_ik_many(
            target_xyz,
            init_betas=seeds_array,
            bounds=bounds,
            lengths_m=np.asarray(lengths_m, dtype=float),
            p_end_local_m=np.asarray(p_end_local_m, dtype=float),
            theta_sign=float(theta_sign),
            max_nfev=int(max_nfev),
            lambda_limit=0.0,
            center_beta=prediction,
            lambda_center=1.0e-3,
        )
        feasible = [solution for solution in solutions if solution.residual_mm <= 1.5]
        reference_beta = previous_beta if previous_beta is not None else parent_beta
        chosen = min(
            feasible if feasible else solutions,
            key=lambda solution: (
                atlas.beta_rms_deg(solution.beta_rad, reference_beta)
                + 0.25 * atlas.beta_rms_deg(solution.beta_rad, parent_beta)
                + 0.10 * atlas.beta_rms_deg(solution.beta_rad, prediction),
                solution.residual_mm,
            ),
        )
        previous_beta = chosen.beta_rad
        record = target_row.to_dict()
        record.update(
            {
                "xyz_residual_mm": float(chosen.residual_mm),
                "tube_success": bool(chosen.residual_mm <= 1.5),
                "parent_offset_id": str(parent_offset_id),
                "inverse_nfev": int(chosen.nfev),
            }
        )
        for idx, column in enumerate(atlas.BETA_COLS):
            record[column] = float(chosen.beta_rad[idx])
        theta = atlas.theta_from_beta_batch(chosen.beta_rad.reshape(1, 6), theta_sign=float(theta_sign))[0]
        for idx, column in enumerate(atlas.THETA_COLS):
            record[column] = float(theta[idx])
        for idx, column in enumerate(atlas.XYZ_COLS):
            record[column] = float(chosen.xyz_m[idx])
        rows.append(record)
    return pd.DataFrame(rows).sort_values("angle_idx").reset_index(drop=True)


def branch_conflict_report(
    frame: pd.DataFrame,
    *,
    voxel_mm: float = 2.0,
    threshold_deg: float = 3.0,
) -> dict[str, Any]:
    """Audit whether nearby Cartesian samples map to incompatible canonical betas."""
    if frame.empty:
        return {
            "rows": 0,
            "occupied_voxels": 0,
            "multi_sample_voxels": 0,
            "conflict_voxels": 0,
            "conflict_voxel_ratio": 1.0,
            "conflict_point_ratio": 1.0,
            "max_voxel_beta_rms_deg": float("inf"),
            "branch_conflict_gate_pass": False,
        }
    scale_m = float(voxel_mm) / 1000.0
    if scale_m <= 0.0:
        raise ValueError("voxel size must be positive")
    xyz = frame[atlas.TARGET_XYZ_COLS].to_numpy(dtype=float)
    voxels = np.floor(xyz / scale_m).astype(np.int64)
    keyed = frame.copy()
    keyed["_voxel_x"] = voxels[:, 0]
    keyed["_voxel_y"] = voxels[:, 1]
    keyed["_voxel_z"] = voxels[:, 2]
    multi_sample = 0
    conflicts = 0
    conflict_points = 0
    maxima: list[float] = []
    for _key, part in keyed.groupby(["_voxel_x", "_voxel_y", "_voxel_z"], sort=False):
        if len(part) < 2:
            continue
        multi_sample += 1
        beta = part[atlas.BETA_COLS].to_numpy(dtype=float)
        delta = beta[:, None, :] - beta[None, :, :]
        distance = np.sqrt(np.mean(np.square(delta), axis=2)) * 180.0 / math.pi
        maximum = float(np.max(distance))
        maxima.append(maximum)
        if maximum > float(threshold_deg):
            conflicts += 1
            conflict_points += int(len(part))
    occupied = int(keyed.groupby(["_voxel_x", "_voxel_y", "_voxel_z"]).ngroups)
    return {
        "rows": int(len(frame)),
        "voxel_mm": float(voxel_mm),
        "branch_conflict_threshold_deg": float(threshold_deg),
        "occupied_voxels": occupied,
        "multi_sample_voxels": int(multi_sample),
        "conflict_voxels": int(conflicts),
        "conflict_voxel_ratio": float(conflicts / max(occupied, 1)),
        "conflict_point_ratio": float(conflict_points / len(frame)),
        "max_voxel_beta_rms_deg": float(max(maxima)) if maxima else 0.0,
        "branch_conflict_gate_pass": bool(conflicts == 0),
    }


def pointwise_task_fingerprint(args: argparse.Namespace, search_report: Mapping[str, Any]) -> str:
    search_dir = Path(args.out_dir) / "01_family_search"
    source_paths = _source_paths(args)
    inputs = {
        "search_task_fingerprint": str(search_report.get("task_fingerprint", "")),
        "primary_ranking": file_sha256(search_dir / "primary_family_ranking.csv"),
        "stretch_ranking": file_sha256(search_dir / "stretch_family_ranking.csv"),
        "family_seeds": file_sha256(search_dir / "family_seeds.csv"),
        "reachability_pool": file_sha256(source_paths["reachability_pool"]),
        "v3_centerline": file_sha256(source_paths["v3_centerline"]),
        "robot_config": file_sha256(Path(args.robot_config)),
    }
    return stable_fingerprint(
        {
            "phase": "pointwise",
            "strategy_version": POINTWISE_STRATEGY_VERSION,
            "protocol": formal_expansion_protocol_report(args)["protocol"],
            "inputs": inputs,
        }
    )


def ensure_pointwise_report(args: argparse.Namespace) -> dict[str, Any]:
    report_path = Path(args.out_dir) / "02_pointwise" / "pointwise_report.json"
    pointwise_dir = report_path.parent
    protocol = formal_expansion_protocol_report(args)
    summary_path = pointwise_dir / "pointwise_radius_summary.csv"
    selected_path = pointwise_dir / "selected_families.csv"
    if (
        not protocol["formal_expansion_protocol_gate_pass"]
        and summary_path.exists()
        and selected_path.exists()
        and not report_path.exists()
    ):
        return {
            "strategy_version": POINTWISE_STRATEGY_VERSION,
            "task_fingerprint": stable_fingerprint(
                {
                    "phase": "pointwise_diagnostic_fixture",
                    "summary": file_sha256(summary_path),
                    "selected": file_sha256(selected_path),
                }
            ),
            **protocol,
            "diagnostic_fixture": True,
        }
    search_report = ensure_search_report(args)
    expected = pointwise_task_fingerprint(args, search_report)
    if report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if phase_cache_is_compatible(
            cached,
            strategy_version=POINTWISE_STRATEGY_VERSION,
            task_fingerprint=expected,
        ):
            return cached
    return phase_pointwise(args)


def phase_pointwise(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "02_pointwise"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "pointwise_report.json"
    search_dir = Path(args.out_dir) / "01_family_search"
    search_report = ensure_search_report(args)
    task_fingerprint = pointwise_task_fingerprint(args, search_report)
    if bool(args.skip_existing) and report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if phase_cache_is_compatible(
            cached,
            strategy_version=POINTWISE_STRATEGY_VERSION,
            task_fingerprint=task_fingerprint,
        ):
            return cached
    allow_partial_cache = _phase_manifest(
        out,
        strategy_version=POINTWISE_STRATEGY_VERSION,
        task_fingerprint=task_fingerprint,
        skip_existing=bool(args.skip_existing),
    )
    primary_ranking = pd.read_csv(search_dir / "primary_family_ranking.csv")
    stretch_ranking = pd.read_csv(search_dir / "stretch_family_ranking.csv")
    seeds = pd.read_csv(search_dir / "family_seeds.csv")
    selected = select_pointwise_families(
        primary_ranking,
        stretch_ranking,
        seeds,
        max_families=int(args.max_pointwise_families),
    )
    selected.to_csv(out / "selected_families.csv", index=False)
    source_paths = _source_paths(args)
    pool = pd.read_parquet(source_paths["reachability_pool"], columns=[*atlas.XYZ_COLS, *atlas.BETA_COLS])
    v3_parent = pd.read_parquet(source_paths["v3_centerline"])
    anchors = sorted(radius for radius in parse_float_csv(args.radius_anchors_mm) if radius <= float(args.stretch_radius_mm) + 1.0e-9)
    budgets = sorted(set(_parse_int_csv(args.pointwise_seed_budgets)))
    if not budgets:
        raise ValueError("pointwise seed budgets must not be empty")
    summary_rows: list[dict[str, Any]] = []
    passing_paths: dict[tuple[str, float], str] = {}
    for _idx, family_row in selected.iterrows():
        family = family_from_mapping(family_row)
        candidate_id = str(family_row["candidate_id"])
        family_dir = out / candidate_id
        family_dir.mkdir(parents=True, exist_ok=True)
        parent_path: pd.DataFrame | None = None
        if candidate_id == "v3_selected" or str(family_row.get("source_family_id", "")) == "v3_selected":
            parent_path = _resample_parent_path(v3_parent, target_count=int(args.coarse_points))
        connected = True
        for radius_mm in anchors:
            radius_dir = family_dir / _radius_slug(radius_mm)
            radius_dir.mkdir(parents=True, exist_ok=True)
            if not connected:
                summary_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "family_id": family.family_id,
                        "radius_mm": float(radius_mm),
                        "executed": False,
                        "pointwise_gate_pass": False,
                        "reason": "previous_anchor_failed",
                    }
                )
                continue
            targets = generate_family_targets(family, radius_mm=float(radius_mm), n_points=int(args.coarse_points))
            targets.to_parquet(radius_dir / "targets.parquet", index=False, compression="zstd")
            combined: list[pd.DataFrame] = []
            final_report: dict[str, Any] | None = None
            final_best: pd.DataFrame | None = None
            for budget in budgets:
                candidates, _best, _budget_report = solve_pointwise_targets_subprocess(
                    targets,
                    pool=pool,
                    robot_config=args.robot_config,
                    seed_budget=int(budget),
                    max_nfev=int(args.max_ik_nfev),
                    workers=int(args.workers),
                    work_dir=radius_dir / f"workers_budget_{int(budget)}",
                    skip_existing=allow_partial_cache,
                    parent_path=parent_path,
                )
                candidates["requested_seed_budget"] = int(budget)
                candidates.to_parquet(radius_dir / f"candidates_budget_{int(budget)}.parquet", index=False, compression="zstd")
                combined.append(candidates)
                all_candidates = pd.concat(combined, ignore_index=True)
                all_candidates = all_candidates.drop_duplicates(["angle_idx", *atlas.BETA_COLS], keep="first")
                final_best = _best_rows(all_candidates)
                final_report = summarize_pointwise_candidates(all_candidates, target_count=len(targets))
                final_report.update(
                    {
                        "candidate_id": candidate_id,
                        "family_id": family.family_id,
                        "radius_mm": float(radius_mm),
                        "executed_seed_budgets": [int(value) for value in budgets if value <= int(budget)],
                        "candidate_rows": int(len(all_candidates)),
                    }
                )
                if bool(final_report["pointwise_gate_pass"]) or not bool(final_report["retry_allowed"]):
                    break
            assert final_report is not None and final_best is not None
            all_candidates.to_parquet(radius_dir / "pointwise_candidates.parquet", index=False, compression="zstd")
            final_best.to_parquet(radius_dir / "best_pointwise_path.parquet", index=False, compression="zstd")
            write_json(radius_dir / "pointwise_report.json", final_report)
            summary_rows.append({**final_report, "executed": True, "reason": "passed" if final_report["pointwise_gate_pass"] else "pointwise_failed"})
            connected = bool(final_report["pointwise_gate_pass"])
            if connected:
                parent_path = final_best
                passing_paths[(candidate_id, float(radius_mm))] = str(radius_dir / "best_pointwise_path.parquet")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "pointwise_radius_summary.csv", index=False)

    def families_at(radius_mm: float) -> list[str]:
        mask = np.isclose(summary["radius_mm"].to_numpy(dtype=float), float(radius_mm), atol=1.0e-8)
        return sorted(summary.loc[mask & summary["pointwise_gate_pass"].fillna(False).astype(bool), "candidate_id"].astype(str).tolist())

    primary_families = families_at(float(args.primary_radius_mm))
    stretch_families = families_at(float(args.stretch_radius_mm))
    report = {
        "strategy_version": POINTWISE_STRATEGY_VERSION,
        "task_fingerprint": task_fingerprint,
        **formal_expansion_protocol_report(args),
        "selected_family_count": int(len(selected)),
        "primary_radius_mm": float(args.primary_radius_mm),
        "stretch_radius_mm": float(args.stretch_radius_mm),
        "primary_passing_families": primary_families,
        "stretch_passing_families": stretch_families,
        "primary_pointwise_gate_pass": bool(primary_families),
        "stretch_pointwise_gate_pass": bool(stretch_families),
        "summary_path": str(out / "pointwise_radius_summary.csv"),
        "passing_paths": {f"{candidate}@{radius:g}": path for (candidate, radius), path in passing_paths.items()},
    }
    write_json(report_path, report)
    return report


def _centerline_report_key(report: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        0.0 if bool(report.get("centerline_gate_pass", False)) else 1.0,
        0.0 if bool(report.get("branch_gate_pass", False)) else 1.0,
        float(report.get("canonical_posture_mean", np.inf)),
        float(report.get("delta_beta_p95_deg", np.inf)),
        float(report.get("residual_p95_mm", np.inf)),
    )


def _augmented_centerline_report(report: Mapping[str, Any], **metadata: Any) -> dict[str, Any]:
    result = dict(report)
    result.update(atlas.evaluate_centerline_gates(result))
    result.update(metadata)
    return result


def _save_path_with_report(path: pd.DataFrame, report: Mapping[str, Any], *, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path.to_parquet(directory / f"{name}.parquet", index=False, compression="zstd")
    write_json(directory / f"{name}.json", dict(report))


def branch_task_fingerprint(args: argparse.Namespace, pointwise_report: Mapping[str, Any]) -> str:
    root = Path(args.out_dir)
    pointwise_dir = root / "02_pointwise"
    search_dir = root / "01_family_search"
    return stable_fingerprint(
        {
            "phase": "branch",
            "strategy_version": BRANCH_SELECTION_STRATEGY_VERSION,
            "repeatability_strategy_version": BRANCH_REPEATABILITY_STRATEGY_VERSION,
            "protocol": formal_expansion_protocol_report(args)["protocol"],
            "pointwise_task_fingerprint": str(pointwise_report.get("task_fingerprint", "")),
            "inputs": {
                "pointwise_summary": _file_hash_or_missing(pointwise_dir / "pointwise_radius_summary.csv"),
                "selected_families": _file_hash_or_missing(pointwise_dir / "selected_families.csv"),
                "primary_ranking": _file_hash_or_missing(search_dir / "primary_family_ranking.csv"),
                "stretch_ranking": _file_hash_or_missing(search_dir / "stretch_family_ranking.csv"),
                "robot_config": _file_hash_or_missing(_arg_path(args, "robot_config", DEFAULT_CONFIG)),
            },
        }
    )


def ensure_branch_report(args: argparse.Namespace) -> dict[str, Any]:
    report_path = Path(args.out_dir) / "03_branch" / "branch_report.json"
    summary_path = report_path.parent / "branch_radius_summary.csv"
    protocol = formal_expansion_protocol_report(args)
    if (
        not protocol["formal_expansion_protocol_gate_pass"]
        and summary_path.exists()
        and not report_path.exists()
    ):
        return {
            "strategy_version": BRANCH_SELECTION_STRATEGY_VERSION,
            "task_fingerprint": stable_fingerprint(
                {"phase": "branch_diagnostic_fixture", "summary": file_sha256(summary_path)}
            ),
            **protocol,
            "diagnostic_fixture": True,
        }
    pointwise_report = ensure_pointwise_report(args)
    expected = branch_task_fingerprint(args, pointwise_report)
    if report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if phase_cache_is_compatible(
            cached,
            strategy_version=BRANCH_SELECTION_STRATEGY_VERSION,
            task_fingerprint=expected,
        ):
            return cached
    return phase_branch(args)


def branch_radius_task_fingerprint(
    *,
    phase_task_fingerprint: str,
    candidate_id: str,
    radius_mm: float,
    pointwise_path: Path,
    parent_centerline_fingerprint: str,
) -> str:
    return stable_fingerprint(
        {
            "phase_task_fingerprint": str(phase_task_fingerprint),
            "candidate_id": str(candidate_id),
            "radius_mm": float(radius_mm),
            "pointwise_path": file_sha256(pointwise_path),
            "parent_centerline_fingerprint": str(parent_centerline_fingerprint),
        }
    )


def branch_radius_cache_is_compatible(
    report: Mapping[str, Any],
    *,
    task_fingerprint: str | None = None,
) -> bool:
    selected_path = str(report.get("selected_path", ""))
    repeatability_compatible = bool(
        selected_path in {"forward", "reverse"}
        or int(report.get("repeatability_strategy_version", 0)) == BRANCH_REPEATABILITY_STRATEGY_VERSION
    )
    fingerprint_compatible = bool(
        task_fingerprint is None or str(report.get("task_fingerprint", "")) == str(task_fingerprint)
    )
    return bool(repeatability_compatible and fingerprint_compatible)


def phase_branch(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "03_branch"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "branch_report.json"
    pointwise_report = ensure_pointwise_report(args)
    task_fingerprint = branch_task_fingerprint(args, pointwise_report)
    if bool(args.skip_existing) and report_path.exists():
        cached_top = json.loads(report_path.read_text(encoding="utf-8"))
        if phase_cache_is_compatible(
            cached_top,
            strategy_version=BRANCH_SELECTION_STRATEGY_VERSION,
            task_fingerprint=task_fingerprint,
        ):
            return cached_top
    allow_partial_cache = _phase_manifest(
        out,
        strategy_version=BRANCH_SELECTION_STRATEGY_VERSION,
        task_fingerprint=task_fingerprint,
        skip_existing=bool(args.skip_existing),
    )
    pointwise_dir = Path(args.out_dir) / "02_pointwise"
    summary_path = pointwise_dir / "pointwise_radius_summary.csv"
    selected_path = pointwise_dir / "selected_families.csv"
    if not summary_path.exists() or not selected_path.exists():
        raise FileNotFoundError("pointwise phase did not materialize the branch inputs")
    pointwise_summary = pd.read_csv(summary_path)
    selected = pd.read_csv(selected_path)
    search_dir = Path(args.out_dir) / "01_family_search"
    primary_search_path = search_dir / "primary_family_ranking.csv"
    stretch_search_path = search_dir / "stretch_family_ranking.csv"
    primary_preference = (
        pd.read_csv(primary_search_path)["candidate_id"].astype(str).tolist() if primary_search_path.exists() else []
    )
    stretch_preference = (
        pd.read_csv(stretch_search_path)["candidate_id"].astype(str).tolist() if stretch_search_path.exists() else []
    )
    selected_ids = select_branch_families(
        pointwise_summary,
        primary_radius_mm=float(args.primary_radius_mm),
        stretch_radius_mm=float(args.stretch_radius_mm),
        max_families=int(args.max_branch_families),
        primary_preference=primary_preference,
        stretch_preference=stretch_preference,
    )
    pd.DataFrame({"candidate_id": selected_ids}).to_csv(out / "selected_branch_families.csv", index=False)
    if not selected_ids:
        report = {
            "strategy_version": BRANCH_SELECTION_STRATEGY_VERSION,
            "task_fingerprint": task_fingerprint,
            **formal_expansion_protocol_report(args),
            "selected_family_count": 0,
            "primary_branch_gate_pass": False,
            "stretch_branch_gate_pass": False,
            "reason": "no_pointwise_family_passed",
        }
        write_json(report_path, report)
        return report

    lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    bounds = atlas.beta_bounds_rad("current")
    anchors = sorted(
        radius
        for radius in parse_float_csv(args.radius_anchors_mm)
        if radius <= float(args.stretch_radius_mm) + 1.0e-9
    )
    summary_rows: list[dict[str, Any]] = []
    centerline_paths: dict[tuple[str, float], str] = {}
    executed_candidate_ids: list[str] = []
    dual_goal_family: str | None = None
    for candidate_id in selected_ids:
        executed_candidate_ids.append(candidate_id)
        family_match = selected[selected["candidate_id"].astype(str).eq(candidate_id)]
        if family_match.empty:
            raise ValueError(f"selected branch family is missing geometry: {candidate_id}")
        family = family_from_mapping(family_match.iloc[0])
        connected = True
        parent_centerline: pd.DataFrame | None = None
        parent_centerline_fingerprint = "none"
        candidate_passing_radii: list[float] = []
        for radius_mm in anchors:
            radius_slug = _radius_slug(radius_mm)
            radius_dir = out / candidate_id / radius_slug
            radius_dir.mkdir(parents=True, exist_ok=True)
            pointwise_mask = (
                pointwise_summary["candidate_id"].astype(str).eq(candidate_id)
                & np.isclose(pointwise_summary["radius_mm"].to_numpy(dtype=float), float(radius_mm), atol=1.0e-8)
            )
            pointwise_row = pointwise_summary.loc[pointwise_mask]
            pointwise_pass = bool(
                not pointwise_row.empty and pointwise_row.iloc[0].get("pointwise_gate_pass", False)
            )
            if not connected or not pointwise_pass:
                summary_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "family_id": family.family_id,
                        "radius_mm": float(radius_mm),
                        "executed": False,
                        "pointwise_gate_pass": pointwise_pass,
                        "branch_robustness_gate_pass": False,
                        "reason": "previous_anchor_failed" if not connected else "pointwise_failed",
                    }
                )
                connected = False
                continue

            pointwise_path = pointwise_dir / candidate_id / radius_slug / "best_pointwise_path.parquet"
            if not pointwise_path.exists():
                raise FileNotFoundError(f"missing pointwise path for {candidate_id}@{radius_mm:g}: {pointwise_path}")
            radius_task_fingerprint = branch_radius_task_fingerprint(
                phase_task_fingerprint=task_fingerprint,
                candidate_id=candidate_id,
                radius_mm=float(radius_mm),
                pointwise_path=pointwise_path,
                parent_centerline_fingerprint=parent_centerline_fingerprint,
            )
            cached_report_path = radius_dir / "branch_radius_report.json"
            cached_centerline_path = radius_dir / "selected_centerline_360.parquet"
            if allow_partial_cache and cached_report_path.exists():
                cached = json.loads(cached_report_path.read_text(encoding="utf-8"))
                if branch_radius_cache_is_compatible(cached, task_fingerprint=radius_task_fingerprint):
                    passed = bool(cached.get("branch_robustness_gate_pass", False) and cached_centerline_path.exists())
                    summary_rows.append(
                        {**cached, "executed": True, "reason": "cached_pass" if passed else "cached_failed"}
                    )
                    connected = passed
                    if passed:
                        parent_centerline = pd.read_parquet(cached_centerline_path)
                        parent_centerline_fingerprint = file_sha256(cached_centerline_path)
                        centerline_paths[(candidate_id, float(radius_mm))] = str(cached_centerline_path)
                        candidate_passing_radii.append(float(radius_mm))
                    continue

            path72 = pd.read_parquet(pointwise_path).sort_values("angle_idx").reset_index(drop=True)
            if parent_centerline is None:
                targets72 = generate_family_targets(family, radius_mm=float(radius_mm), n_points=int(args.coarse_points))
                optimized72, report72_raw = atlas.optimize_cyclic_trajectory(
                    targets=targets72,
                    initial_beta=periodic_resample_beta(path72, target_count=int(args.coarse_points)),
                    bounds=bounds,
                    lengths_m=lengths_m,
                    p_end_local_m=p_end_local_m,
                    theta_sign=theta_sign,
                    max_nfev=int(args.max_opt_nfev),
                    stop_on_centerline_gate=True,
                )
                report72 = _augmented_centerline_report(
                    report72_raw,
                    source="pointwise_optimized_72",
                    resolution_points=int(len(optimized72)),
                )
                _save_path_with_report(optimized72, report72, directory=radius_dir, name="optimized_72")
                start_beta = optimized72.iloc[0][atlas.BETA_COLS].to_numpy(dtype=float)
                start_source = "pointwise_optimized_72"
            else:
                start_beta = parent_centerline.sort_values("angle_idx").iloc[0][atlas.BETA_COLS].to_numpy(dtype=float)
                start_source = "previous_radius_centerline"
                write_json(
                    radius_dir / "radial_start.json",
                    {
                        "source": start_source,
                        "parent_radius_mm": float(candidate_passing_radii[-1]),
                        "start_beta": start_beta.tolist(),
                    },
                )

            targets360 = generate_family_targets(family, radius_mm=float(radius_mm), n_points=int(args.final_points))
            continuation_paths: dict[str, pd.DataFrame] = {}
            continuation_reports: dict[str, dict[str, Any]] = {}
            repeat_paths: dict[str, pd.DataFrame] = {}
            use_subprocess_branch = bool(int(args.final_points) >= 360 and int(args.workers) > 1)
            if use_subprocess_branch:
                targets360_path = radius_dir / "targets_360.parquet"
                targets360.to_parquet(targets360_path, index=False, compression="zstd")
                task_paths: list[Path] = []
                for direction in ("forward", "reverse"):
                    for repeat_rank, suffix in enumerate(("", "_repeat")):
                        name = f"continuation_{direction}{suffix}_360"
                        task_path = radius_dir / f"worker_{name}.json"
                        write_json(
                            task_path,
                            {
                                "kind": "branch_continuation",
                                "task_id": name,
                                "robot_config": str(args.robot_config),
                                "targets_path": str(targets360_path),
                                "start_beta": start_beta.tolist(),
                                "direction": direction,
                                "source": f"continuation_{direction}",
                                "max_nfev": int(args.max_ik_nfev),
                                "output_path": str(radius_dir / f"{name}.parquet"),
                                "report_path": str(radius_dir / f"{name}.json"),
                                "repeat_rank": repeat_rank,
                            },
                        )
                        task_paths.append(task_path)
                _execute_worker_tasks(task_paths, workers=min(int(args.workers), 4))
                for direction in ("forward", "reverse"):
                    continuation_paths[direction] = pd.read_parquet(radius_dir / f"continuation_{direction}_360.parquet")
                    continuation_reports[direction] = json.loads(
                        (radius_dir / f"continuation_{direction}_360.json").read_text(encoding="utf-8")
                    )
                    repeat_paths[direction] = pd.read_parquet(radius_dir / f"continuation_{direction}_repeat_360.parquet")
            else:
                for direction in ("forward", "reverse"):
                    lifted, raw = atlas.continuation_lift(
                        targets=targets360,
                        start_beta=start_beta,
                        bounds=bounds,
                        lengths_m=lengths_m,
                        p_end_local_m=p_end_local_m,
                        theta_sign=theta_sign,
                        direction=direction,
                        method="warm",
                        lambda_center=1.0e-3,
                        lambda_limit=0.0,
                        max_nfev=int(args.max_ik_nfev),
                    )
                    augmented = _augmented_centerline_report(
                        raw,
                        source=f"continuation_{direction}",
                        direction=direction,
                        resolution_points=int(len(lifted)),
                    )
                    continuation_paths[direction] = lifted
                    continuation_reports[direction] = augmented
                    _save_path_with_report(lifted, augmented, directory=radius_dir, name=f"continuation_{direction}_360")
            forward_reverse = atlas.branch_reproducibility_report(
                continuation_paths["forward"],
                continuation_paths["reverse"],
                threshold_deg=1.0,
            )

            selected_direction = min(continuation_reports, key=lambda name: _centerline_report_key(continuation_reports[name]))
            optimized_candidates: list[tuple[str, pd.DataFrame, dict[str, Any]]] = []
            optimization_initials: dict[str, np.ndarray] = {}
            continuation_already_passes = any(
                bool(report.get("centerline_gate_pass", False)) for report in continuation_reports.values()
            )
            if bool(forward_reverse.get("reproducible", False)) and not continuation_already_passes:
                optimization_inputs: list[tuple[str, np.ndarray]] = [
                    (f"continuation_{selected_direction}", continuation_paths[selected_direction][atlas.BETA_COLS].to_numpy(dtype=float))
                ]
                if parent_centerline is not None and len(parent_centerline) == len(targets360):
                    optimization_inputs.append(
                        ("radial_parent", parent_centerline.sort_values("angle_idx")[atlas.BETA_COLS].to_numpy(dtype=float))
                    )
                for source_name, initial_beta in optimization_inputs:
                    optimized_name = f"optimized_{source_name}"
                    optimization_initials[optimized_name] = np.asarray(initial_beta, dtype=float).copy()
                    optimized, raw = atlas.optimize_cyclic_trajectory(
                        targets=targets360,
                        initial_beta=optimization_initials[optimized_name].copy(),
                        bounds=bounds,
                        lengths_m=lengths_m,
                        p_end_local_m=p_end_local_m,
                        theta_sign=theta_sign,
                        max_nfev=int(args.max_opt_nfev),
                        stop_on_centerline_gate=True,
                    )
                    augmented = _augmented_centerline_report(
                        raw,
                        source=optimized_name,
                        resolution_points=int(len(optimized)),
                    )
                    optimized_candidates.append((optimized_name, optimized, augmented))
                    _save_path_with_report(optimized, augmented, directory=radius_dir, name=f"{optimized_name}_360")

            path_candidates: list[tuple[str, pd.DataFrame, dict[str, Any]]] = [
                (name, continuation_paths[name], continuation_reports[name]) for name in ("forward", "reverse")
            ]
            path_candidates.extend(optimized_candidates)
            selected_name, selected_centerline, selected_report = min(
                path_candidates,
                key=lambda item: _centerline_report_key(item[2]),
            )
            if bool(forward_reverse.get("reproducible", False)):
                if selected_name in continuation_paths and selected_name in repeat_paths:
                    repeat_path = repeat_paths[selected_name]
                    repeat_algorithm = f"continuation_{selected_name}"
                elif selected_name in continuation_paths:
                    repeat_path, _repeat_raw = atlas.continuation_lift(
                        targets=targets360,
                        start_beta=start_beta,
                        bounds=bounds,
                        lengths_m=lengths_m,
                        p_end_local_m=p_end_local_m,
                        theta_sign=theta_sign,
                        direction=selected_name,
                        method="warm",
                        lambda_center=1.0e-3,
                        lambda_limit=0.0,
                        max_nfev=int(args.max_ik_nfev),
                    )
                    repeat_algorithm = f"continuation_{selected_name}"
                elif selected_name in optimization_initials:
                    repeat_path, repeat_raw = atlas.optimize_cyclic_trajectory(
                        targets=targets360,
                        initial_beta=optimization_initials[selected_name].copy(),
                        bounds=bounds,
                        lengths_m=lengths_m,
                        p_end_local_m=p_end_local_m,
                        theta_sign=theta_sign,
                        max_nfev=int(args.max_opt_nfev),
                        stop_on_centerline_gate=True,
                    )
                    repeat_report = _augmented_centerline_report(
                        repeat_raw,
                        source=f"{selected_name}_repeat",
                        resolution_points=int(len(repeat_path)),
                    )
                    _save_path_with_report(
                        repeat_path,
                        repeat_report,
                        directory=radius_dir,
                        name=f"{selected_name}_repeat_360",
                    )
                    repeat_algorithm = selected_name
                else:
                    raise AssertionError(f"selected branch path has no repeatable algorithm: {selected_name}")
                repeatability = atlas.branch_reproducibility_report(
                    selected_centerline,
                    repeat_path,
                    threshold_deg=1.0e-6,
                )
                repeatability["algorithm"] = repeat_algorithm
            else:
                repeatability = {
                    "rows": 0,
                    "branch_diff_p95_deg": float("inf"),
                    "reproducible": False,
                    "reason": "forward_reverse_gate_failed",
                }
            robust_report = evaluate_branch_robustness(
                centerline_report=selected_report,
                forward_reverse=forward_reverse,
                repeatability=repeatability,
            )
            robust_report.update(
                {
                    "candidate_id": candidate_id,
                    "family_id": family.family_id,
                    "radius_mm": float(radius_mm),
                    "task_fingerprint": radius_task_fingerprint,
                    "selected_path": selected_name,
                    "selected_continuation_direction": selected_direction,
                    "repeatability_strategy_version": BRANCH_REPEATABILITY_STRATEGY_VERSION,
                    "start_source": start_source,
                    "forward_reverse": forward_reverse,
                    "deterministic_repeatability": repeatability,
                }
            )
            passed = bool(robust_report["branch_robustness_gate_pass"])
            if passed:
                selected_centerline.to_parquet(cached_centerline_path, index=False, compression="zstd")
                parent_centerline = selected_centerline
                parent_centerline_fingerprint = file_sha256(cached_centerline_path)
                centerline_paths[(candidate_id, float(radius_mm))] = str(cached_centerline_path)
                candidate_passing_radii.append(float(radius_mm))
            write_json(cached_report_path, robust_report)
            summary_rows.append(
                {
                    **{key: value for key, value in robust_report.items() if not isinstance(value, (dict, list))},
                    "executed": True,
                    "pointwise_gate_pass": True,
                    "reason": "passed" if passed else "branch_failed",
                }
            )
            connected = passed
        if family_covers_both_goals(
            candidate_passing_radii,
            primary_radius_mm=float(args.primary_radius_mm),
            stretch_radius_mm=float(args.stretch_radius_mm),
        ):
            if dual_goal_family is None:
                dual_goal_family = candidate_id

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "branch_radius_summary.csv", index=False)
    all_selected_families_executed = executed_candidate_ids == selected_ids
    if not all_selected_families_executed:
        raise AssertionError(
            f"formal branch validation did not execute every selected family: "
            f"selected={selected_ids}, executed={executed_candidate_ids}"
        )

    def passing_at(radius_mm: float) -> list[str]:
        if summary.empty:
            return []
        mask = np.isclose(summary["radius_mm"].to_numpy(dtype=float), float(radius_mm), atol=1.0e-8)
        return sorted(
            summary.loc[
                mask & summary["branch_robustness_gate_pass"].fillna(False).astype(bool),
                "candidate_id",
            ].astype(str).tolist()
        )

    primary_families = passing_at(float(args.primary_radius_mm))
    stretch_families = passing_at(float(args.stretch_radius_mm))
    report = {
        "strategy_version": BRANCH_SELECTION_STRATEGY_VERSION,
        "task_fingerprint": task_fingerprint,
        **formal_expansion_protocol_report(args),
        "selection_strategy_version": BRANCH_SELECTION_STRATEGY_VERSION,
        "max_branch_families": int(args.max_branch_families),
        "selected_family_count": int(len(selected_ids)),
        "selected_families": selected_ids,
        "executed_families": executed_candidate_ids,
        "all_selected_families_executed": all_selected_families_executed,
        "dual_goal_family": dual_goal_family,
        "primary_radius_mm": float(args.primary_radius_mm),
        "stretch_radius_mm": float(args.stretch_radius_mm),
        "radius_anchors_mm": anchors,
        "primary_passing_families": primary_families,
        "stretch_passing_families": stretch_families,
        "primary_branch_gate_pass": bool(primary_families),
        "stretch_branch_gate_pass": bool(stretch_families),
        "summary_path": str(out / "branch_radius_summary.csv"),
        "centerline_paths": {
            f"{candidate}@{radius:g}": path for (candidate, radius), path in centerline_paths.items()
        },
    }
    write_json(report_path, report)
    return report


def _offset_file_id(pair: tuple[float, float]) -> str:
    return f"n1_{pair[0]:g}_n2_{pair[1]:g}".replace("-", "m").replace(".", "p")


def tube_task_fingerprint(args: argparse.Namespace, branch_report: Mapping[str, Any]) -> str:
    branch_dir = Path(args.out_dir) / "03_branch"
    summary_path = branch_dir / "branch_radius_summary.csv"
    centerlines: dict[str, str] = {}
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "branch_robustness_gate_pass" in summary:
            passing = summary[summary["branch_robustness_gate_pass"].fillna(False).astype(bool)]
            for _idx, row in passing.iterrows():
                candidate_id = str(row["candidate_id"])
                radius_mm = float(row["radius_mm"])
                path = branch_dir / candidate_id / _radius_slug(radius_mm) / "selected_centerline_360.parquet"
                centerlines[f"{candidate_id}@{radius_mm:g}"] = _file_hash_or_missing(path)
    return stable_fingerprint(
        {
            "phase": "tube",
            "strategy_version": TUBE_STRATEGY_VERSION,
            "protocol": formal_expansion_protocol_report(args)["protocol"],
            "branch_task_fingerprint": str(branch_report.get("task_fingerprint", "")),
            "inputs": {
                "branch_summary": _file_hash_or_missing(summary_path),
                "centerlines": centerlines,
                "robot_config": _file_hash_or_missing(_arg_path(args, "robot_config", DEFAULT_CONFIG)),
            },
        }
    )


def ensure_tube_report(args: argparse.Namespace) -> dict[str, Any]:
    report_path = Path(args.out_dir) / "04_tube" / "tube_report.json"
    summary_path = report_path.parent / "tube_radius_summary.csv"
    protocol = formal_expansion_protocol_report(args)
    if (
        not protocol["formal_expansion_protocol_gate_pass"]
        and summary_path.exists()
        and not report_path.exists()
    ):
        return {
            "strategy_version": TUBE_STRATEGY_VERSION,
            "task_fingerprint": stable_fingerprint(
                {"phase": "tube_diagnostic_fixture", "summary": file_sha256(summary_path)}
            ),
            **protocol,
            "diagnostic_fixture": True,
        }
    branch_report = ensure_branch_report(args)
    expected = tube_task_fingerprint(args, branch_report)
    if report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if phase_cache_is_compatible(
            cached,
            strategy_version=TUBE_STRATEGY_VERSION,
            task_fingerprint=expected,
        ):
            return cached
    return phase_tube(args)


def tube_radius_task_fingerprint(
    *,
    phase_task_fingerprint: str,
    candidate_id: str,
    radius_mm: float,
    centerline_path: Path,
) -> str:
    return stable_fingerprint(
        {
            "phase_task_fingerprint": str(phase_task_fingerprint),
            "candidate_id": str(candidate_id),
            "radius_mm": float(radius_mm),
            "centerline": file_sha256(centerline_path),
        }
    )


def tube_curve_task_fingerprint(
    *,
    radius_task_fingerprint: str,
    pair: tuple[float, float],
    parent_pair: tuple[float, float],
    parent_curve_path: Path,
    max_nfev: int,
) -> str:
    return stable_fingerprint(
        {
            "radius_task_fingerprint": str(radius_task_fingerprint),
            "pair": [float(pair[0]), float(pair[1])],
            "parent_pair": [float(parent_pair[0]), float(parent_pair[1])],
            "parent_curve": file_sha256(parent_curve_path),
            "max_nfev": int(max_nfev),
        }
    )


def phase_tube(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "04_tube"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "tube_report.json"
    offsets = validate_formal_tube_offsets(args.tube_offsets_mm)
    branch_report = ensure_branch_report(args)
    task_fingerprint = tube_task_fingerprint(args, branch_report)
    if bool(args.skip_existing) and report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if phase_cache_is_compatible(
            cached,
            strategy_version=TUBE_STRATEGY_VERSION,
            task_fingerprint=task_fingerprint,
        ):
            return cached
    allow_partial_cache = _phase_manifest(
        out,
        strategy_version=TUBE_STRATEGY_VERSION,
        task_fingerprint=task_fingerprint,
        skip_existing=bool(args.skip_existing),
    )
    branch_dir = Path(args.out_dir) / "03_branch"
    branch_summary_path = branch_dir / "branch_radius_summary.csv"
    if not branch_summary_path.exists():
        raise FileNotFoundError("branch phase did not materialize the tube inputs")
    branch_summary = pd.read_csv(branch_summary_path)
    lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    offset_pairs = [(float(left), float(right)) for left in offsets for right in offsets]
    offset_pairs.sort(key=lambda pair: (math.hypot(pair[0], pair[1]), math.atan2(pair[1], pair[0])))
    center_pair = (0.0, 0.0)
    if center_pair not in offset_pairs:
        raise ValueError("tube offset grid must contain the centerline offset (0, 0)")
    summary_rows: list[dict[str, Any]] = []
    tube_paths: dict[tuple[str, float], str] = {}
    candidate_ids = sorted(branch_summary["candidate_id"].astype(str).unique().tolist())
    for candidate_id in candidate_ids:
        candidate_rows = branch_summary[branch_summary["candidate_id"].astype(str).eq(candidate_id)].sort_values("radius_mm")
        connected = True
        for _row_idx, branch_row in candidate_rows.iterrows():
            radius_mm = float(branch_row["radius_mm"])
            branch_pass = bool(branch_row.get("branch_robustness_gate_pass", False))
            radius_dir = out / candidate_id / _radius_slug(radius_mm)
            radius_dir.mkdir(parents=True, exist_ok=True)
            if not connected or not branch_pass:
                summary_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "family_id": str(branch_row.get("family_id", candidate_id)),
                        "radius_mm": radius_mm,
                        "executed": False,
                        "branch_robustness_gate_pass": branch_pass,
                        "tube_gate_pass": False,
                        "reason": "previous_anchor_failed" if not connected else "branch_failed",
                    }
                )
                connected = False
                continue
            quality_path = radius_dir / "tube_quality_report.json"
            tube_small_path = radius_dir / "tube_small.parquet"
            centerline_path = branch_dir / candidate_id / _radius_slug(radius_mm) / "selected_centerline_360.parquet"
            if not centerline_path.exists():
                raise FileNotFoundError(f"missing robust centerline for tube: {centerline_path}")
            radius_task_fingerprint = tube_radius_task_fingerprint(
                phase_task_fingerprint=task_fingerprint,
                candidate_id=candidate_id,
                radius_mm=radius_mm,
                centerline_path=centerline_path,
            )
            if allow_partial_cache and quality_path.exists():
                cached = json.loads(quality_path.read_text(encoding="utf-8"))
                if str(cached.get("task_fingerprint", "")) == radius_task_fingerprint:
                    passed = bool(cached.get("tube_gate_pass", False) and tube_small_path.exists())
                    summary_rows.append({**cached, "executed": True, "reason": "cached_pass" if passed else "cached_failed"})
                    connected = passed
                    if passed:
                        tube_paths[(candidate_id, radius_mm)] = str(tube_small_path)
                    continue
            centerline = pd.read_parquet(centerline_path).sort_values("angle_idx").reset_index(drop=True)
            tube_targets = atlas.make_normal_tube_targets(centerline, offsets_mm=offsets)
            tube_targets.to_parquet(radius_dir / "tube_targets.parquet", index=False, compression="zstd")
            center_beta = centerline[atlas.BETA_COLS].to_numpy(dtype=float)
            center_xyz = centerline[atlas.XYZ_COLS].to_numpy(dtype=float)
            pinv = np.asarray(
                [
                    atlas.weighted_damped_pinv(
                        atlas.numerical_jacobian_beta(
                            beta,
                            lengths_m=lengths_m,
                            p_end_local_m=p_end_local_m,
                            theta_sign=theta_sign,
                        ),
                        damping=1.0e-3,
                        weights=np.asarray([4, 4, 2, 2, 1, 1], dtype=float),
                    )
                    for beta in center_beta
                ],
                dtype=float,
            )
            np.save(radius_dir / "weighted_pinv.npy", pinv)
            curves_dir = radius_dir / "curves"
            curves_dir.mkdir(parents=True, exist_ok=True)
            center_targets = tube_targets[
                np.isclose(tube_targets["delta_n1_mm"].to_numpy(dtype=float), 0.0)
                & np.isclose(tube_targets["delta_n2_mm"].to_numpy(dtype=float), 0.0)
            ].sort_values("angle_idx").reset_index(drop=True)
            center_curve = center_targets.copy()
            for idx, column in enumerate(atlas.BETA_COLS):
                center_curve[column] = center_beta[:, idx]
            theta = atlas.theta_from_beta_batch(center_beta, theta_sign=theta_sign)
            for idx, column in enumerate(atlas.THETA_COLS):
                center_curve[column] = theta[:, idx]
            for idx, column in enumerate(atlas.XYZ_COLS):
                center_curve[column] = center_xyz[:, idx]
            center_curve["xyz_residual_mm"] = np.linalg.norm(
                center_xyz - center_curve[atlas.TARGET_XYZ_COLS].to_numpy(dtype=float), axis=1
            ) * 1000.0
            center_curve["tube_success"] = center_curve["xyz_residual_mm"].le(1.5)
            center_curve["parent_offset_id"] = "centerline"
            center_curve["inverse_nfev"] = 0
            center_path = curves_dir / f"{_offset_file_id(center_pair)}.parquet"
            center_curve.to_parquet(center_path, index=False, compression="zstd")
            solved_curves: dict[tuple[float, float], pd.DataFrame] = {center_pair: center_curve}
            solved_curve_paths: dict[tuple[float, float], Path] = {center_pair: center_path}
            use_subprocess_tube = bool(len(centerline) >= 360 and int(args.workers) > 1)
            curve_targets_dir = radius_dir / "curve_targets"
            worker_tasks_dir = radius_dir / "worker_tasks"
            curve_targets_dir.mkdir(parents=True, exist_ok=True)
            worker_tasks_dir.mkdir(parents=True, exist_ok=True)

            radius_groups: dict[float, list[tuple[float, float]]] = {}
            for pair in offset_pairs:
                if pair != center_pair:
                    radius_groups.setdefault(round(math.hypot(pair[0], pair[1]), 8), []).append(pair)
            for offset_radius in sorted(radius_groups):
                jobs: list[tuple[tuple[float, float], tuple[float, float], pd.DataFrame, Path]] = []
                for pair in radius_groups[offset_radius]:
                    curve_path = curves_dir / f"{_offset_file_id(pair)}.parquet"
                    smaller = [
                        old_pair
                        for old_pair in solved_curves
                        if math.hypot(old_pair[0], old_pair[1]) < offset_radius - 1.0e-12
                    ]
                    parent_pair = min(
                        smaller,
                        key=lambda old: math.hypot(old[0] - pair[0], old[1] - pair[1]),
                    ) if smaller else center_pair
                    curve_targets = tube_targets[
                        np.isclose(tube_targets["delta_n1_mm"].to_numpy(dtype=float), pair[0])
                        & np.isclose(tube_targets["delta_n2_mm"].to_numpy(dtype=float), pair[1])
                    ].sort_values("angle_idx").reset_index(drop=True)
                    curve_report_path = curve_path.with_suffix(".json")
                    curve_task_fingerprint = tube_curve_task_fingerprint(
                        radius_task_fingerprint=radius_task_fingerprint,
                        pair=pair,
                        parent_pair=parent_pair,
                        parent_curve_path=solved_curve_paths[parent_pair],
                        max_nfev=int(args.max_ik_nfev),
                    )
                    if allow_partial_cache and curve_path.exists() and curve_report_path.exists():
                        cached_curve = json.loads(curve_report_path.read_text(encoding="utf-8"))
                        if str(cached_curve.get("task_fingerprint", "")) == curve_task_fingerprint:
                            solved_curves[pair] = pd.read_parquet(curve_path)
                            solved_curve_paths[pair] = curve_path
                            continue
                    jobs.append((pair, parent_pair, curve_targets, curve_path))

                def solve_job(job: tuple[tuple[float, float], tuple[float, float], pd.DataFrame, Path]) -> tuple[tuple[float, float], pd.DataFrame, Path]:
                    pair, parent_pair, curve_targets, curve_path = job
                    curve = solve_tube_curve(
                        curve_targets,
                        centerline=centerline,
                        parent_curve=solved_curves[parent_pair],
                        weighted_pinv=pinv,
                        lengths_m=lengths_m,
                        p_end_local_m=p_end_local_m,
                        theta_sign=theta_sign,
                        max_nfev=int(args.max_ik_nfev),
                        parent_offset_id=f"n1_{parent_pair[0]:g}_n2_{parent_pair[1]:g}",
                    )
                    return pair, curve, curve_path

                if jobs:
                    if use_subprocess_tube:
                        task_paths: list[Path] = []
                        for pair, parent_pair, curve_targets, curve_path in jobs:
                            file_id = _offset_file_id(pair)
                            curve_task_fingerprint = tube_curve_task_fingerprint(
                                radius_task_fingerprint=radius_task_fingerprint,
                                pair=pair,
                                parent_pair=parent_pair,
                                parent_curve_path=solved_curve_paths[parent_pair],
                                max_nfev=int(args.max_ik_nfev),
                            )
                            targets_path = curve_targets_dir / f"{file_id}.parquet"
                            curve_targets.to_parquet(targets_path, index=False, compression="zstd")
                            task_path = worker_tasks_dir / f"{file_id}.json"
                            write_json(
                                task_path,
                                {
                                    "kind": "tube_curve",
                                    "task_id": file_id,
                                    "robot_config": str(args.robot_config),
                                    "centerline_path": str(centerline_path),
                                    "curve_targets_path": str(targets_path),
                                    "parent_curve_path": str(solved_curve_paths[parent_pair]),
                                    "pinv_path": str(radius_dir / "weighted_pinv.npy"),
                                    "max_nfev": int(args.max_ik_nfev),
                                    "parent_offset_id": f"n1_{parent_pair[0]:g}_n2_{parent_pair[1]:g}",
                                    "offset_id": f"n1_{pair[0]:g}_n2_{pair[1]:g}",
                                    "task_fingerprint": curve_task_fingerprint,
                                    "output_path": str(curve_path),
                                    "report_path": str(curve_path.with_suffix(".json")),
                                },
                            )
                            task_paths.append(task_path)
                        _execute_worker_tasks(task_paths, workers=int(args.workers))
                        solved_jobs = [
                            (pair, pd.read_parquet(curve_path), curve_path)
                            for pair, _parent_pair, _curve_targets, curve_path in jobs
                        ]
                    elif int(args.workers) <= 1:
                        solved_jobs = [solve_job(job) for job in jobs]
                    else:
                        with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
                            solved_jobs = list(executor.map(solve_job, jobs))
                    for pair, curve, curve_path in solved_jobs:
                        if not use_subprocess_tube:
                            parent_pair = next(
                                parent
                                for job_pair, parent, _curve_targets, job_path in jobs
                                if job_pair == pair and job_path == curve_path
                            )
                            curve_task_fingerprint = tube_curve_task_fingerprint(
                                radius_task_fingerprint=radius_task_fingerprint,
                                pair=pair,
                                parent_pair=parent_pair,
                                parent_curve_path=solved_curve_paths[parent_pair],
                                max_nfev=int(args.max_ik_nfev),
                            )
                            curve.to_parquet(curve_path, index=False, compression="zstd")
                            write_json(
                                curve_path.with_suffix(".json"),
                                {
                                    "task_fingerprint": curve_task_fingerprint,
                                    "offset_id": f"n1_{pair[0]:g}_n2_{pair[1]:g}",
                                    "rows": int(len(curve)),
                                    "success_ratio": float(curve["tube_success"].mean()),
                                    "residual_p95_mm": float(np.percentile(curve["xyz_residual_mm"], 95)),
                                    "residual_max_mm": float(curve["xyz_residual_mm"].max()),
                                },
                            )
                        solved_curves[pair] = curve
                        solved_curve_paths[pair] = curve_path

            tube = pd.concat([solved_curves[pair] for pair in offset_pairs], ignore_index=True)
            tube.to_parquet(radius_dir / "tube_attempt.parquet", index=False, compression="zstd")
            residual = tube["xyz_residual_mm"].to_numpy(dtype=float)
            local = atlas.local_beta_consistency_report(
                tube,
                radius_mm=10.0,
                max_neighbors=None,
                multi_branch_threshold_deg=3.0,
            )
            quality: dict[str, Any] = {
                "task_fingerprint": radius_task_fingerprint,
                "candidate_id": candidate_id,
                "family_id": str(branch_row.get("family_id", candidate_id)),
                "radius_mm": radius_mm,
                "rows": int(len(tube)),
                "expected_rows": int(len(tube_targets)),
                "normal_grid_size": int(len(offset_pairs)),
                "normal_grid_offsets_mm": offsets,
                "target_success_ratio": float(tube["tube_success"].mean()),
                "normal_grid_coverage_ratio": float(len(tube) / max(len(tube_targets), 1)),
                "residual_p95_mm": float(np.percentile(residual, 95)),
                "residual_max_mm": float(np.max(residual)),
                **local,
            }
            quality["tube_gate_pass"] = tube_gate_pass(quality)
            passed = bool(quality["tube_gate_pass"])
            if passed:
                tube.to_parquet(tube_small_path, index=False, compression="zstd")
                tube_paths[(candidate_id, radius_mm)] = str(tube_small_path)
            write_json(quality_path, quality)
            summary_rows.append(
                {
                    **{key: value for key, value in quality.items() if not isinstance(value, (dict, list))},
                    "executed": True,
                    "branch_robustness_gate_pass": True,
                    "reason": "passed" if passed else "tube_failed",
                }
            )
            connected = passed

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "tube_radius_summary.csv", index=False)

    def passing_at(radius_mm: float) -> list[str]:
        if summary.empty:
            return []
        mask = np.isclose(summary["radius_mm"].to_numpy(dtype=float), float(radius_mm), atol=1.0e-8)
        return sorted(
            summary.loc[mask & summary["tube_gate_pass"].fillna(False).astype(bool), "candidate_id"].astype(str).tolist()
        )

    primary_families = passing_at(float(args.primary_radius_mm))
    stretch_families = passing_at(float(args.stretch_radius_mm))
    report = {
        "strategy_version": TUBE_STRATEGY_VERSION,
        "task_fingerprint": task_fingerprint,
        **formal_expansion_protocol_report(args),
        "primary_radius_mm": float(args.primary_radius_mm),
        "stretch_radius_mm": float(args.stretch_radius_mm),
        "primary_passing_families": primary_families,
        "stretch_passing_families": stretch_families,
        "primary_tube_gate_pass": bool(primary_families),
        "stretch_tube_gate_pass": bool(stretch_families),
        "summary_path": str(out / "tube_radius_summary.csv"),
        "tube_paths": {f"{candidate}@{radius:g}": path for (candidate, radius), path in tube_paths.items()},
    }
    write_json(report_path, report)
    return report


def _formal_family_coverage_from_artifacts(
    args: argparse.Namespace,
) -> tuple[dict[str, Path], dict[str, Any]]:
    root = Path(args.out_dir)
    paths = {
        "pointwise_report": root / "02_pointwise" / "pointwise_report.json",
        "pointwise_selection": root / "02_pointwise" / "selected_families.csv",
        "branch_report": root / "03_branch" / "branch_report.json",
    }
    pointwise_report = (
        json.loads(paths["pointwise_report"].read_text(encoding="utf-8"))
        if paths["pointwise_report"].is_file()
        else {}
    )
    branch_report = (
        json.loads(paths["branch_report"].read_text(encoding="utf-8"))
        if paths["branch_report"].is_file()
        else {}
    )
    pointwise_selected_ids: list[str] = []
    if paths["pointwise_selection"].is_file():
        pointwise_selection = pd.read_csv(paths["pointwise_selection"])
        if "candidate_id" in pointwise_selection:
            pointwise_selected_ids = pointwise_selection["candidate_id"].astype(str).tolist()
    coverage = formal_family_coverage_report(
        pointwise_report,
        branch_report,
        pointwise_selected_families=pointwise_selected_ids,
    )
    return paths, coverage


def _formal_family_coverage_fields(
    paths: Mapping[str, Path],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "formal_family_coverage_gate_pass": bool(
            coverage.get("formal_family_coverage_gate_pass", False)
        ),
        "formal_family_coverage_checks": dict(coverage.get("checks", {})),
        "formal_family_coverage": dict(coverage.get("evidence", {})),
        "formal_family_coverage_fingerprint": str(
            coverage.get("formal_family_coverage_fingerprint", "")
        ),
    }
    for name, path in paths.items():
        payload[f"{name}_path"] = str(path.resolve())
        payload[f"{name}_sha256"] = file_sha256(path) if path.is_file() else None
        payload[f"{name}_bytes"] = int(path.stat().st_size) if path.is_file() else 0
    return payload


def formal_dataset_claims_allowed(
    args: argparse.Namespace,
    *,
    dataset_gate: bool,
    tube_report: Mapping[str, Any],
    family_coverage_report: Mapping[str, Any],
    artifact_binding_complete: bool,
) -> bool:
    return bool(
        dataset_gate
        and formal_expansion_protocol_report(args)["formal_expansion_protocol_gate_pass"]
        and tube_report.get("formal_expansion_protocol_gate_pass", False)
        and family_coverage_report.get("formal_family_coverage_gate_pass", False)
        and artifact_binding_complete
    )


def dataset_task_fingerprint(args: argparse.Namespace, tube_report: Mapping[str, Any]) -> str:
    tube_dir = Path(args.out_dir) / "04_tube"
    summary_path = tube_dir / "tube_radius_summary.csv"
    family_coverage_paths, family_coverage = _formal_family_coverage_from_artifacts(args)
    tube_inputs: dict[str, str] = {}
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "tube_gate_pass" in summary:
            passing = summary[summary["tube_gate_pass"].fillna(False).astype(bool)]
            for _idx, row in passing.iterrows():
                candidate_id = str(row["candidate_id"])
                radius_mm = float(row["radius_mm"])
                path = tube_dir / candidate_id / _radius_slug(radius_mm) / "tube_small.parquet"
                tube_inputs[f"{candidate_id}@{radius_mm:g}"] = _file_hash_or_missing(path)
    return stable_fingerprint(
        {
            "phase": "dataset",
            "strategy_version": DATASET_STRATEGY_VERSION,
            "protocol": formal_expansion_protocol_report(args)["protocol"],
            "tube_task_fingerprint": str(tube_report.get("task_fingerprint", "")),
            "inputs": {
                "tube_summary": _file_hash_or_missing(summary_path),
                "passing_tubes": tube_inputs,
                "robot_config": _file_hash_or_missing(_arg_path(args, "robot_config", DEFAULT_CONFIG)),
                **{
                    name: _file_hash_or_missing(path)
                    for name, path in family_coverage_paths.items()
                },
            },
            "formal_family_coverage_fingerprint": family_coverage[
                "formal_family_coverage_fingerprint"
            ],
        }
    )


def phase_dataset(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "05_dataset"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "dataset_report.json"
    final_dataset_path = out / "true_ellipse_family_tubes_v5.parquet"
    manifest_path = out / "trajectory_manifest.csv"
    robot_config_path = _arg_path(args, "robot_config", DEFAULT_CONFIG).resolve()
    tube_report = ensure_tube_report(args)
    family_coverage_paths, family_coverage = _formal_family_coverage_from_artifacts(args)
    family_coverage_fields = _formal_family_coverage_fields(
        family_coverage_paths,
        family_coverage,
    )
    task_fingerprint = dataset_task_fingerprint(args, tube_report)
    if bool(args.skip_existing) and report_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        dataset_cache_matches = bool(
            cached.get("dataset_gate_pass", False)
            and final_dataset_path.exists()
            and str(cached.get("dataset_path", "")) == str(final_dataset_path.resolve())
            and str(cached.get("dataset_sha256", "")) == file_sha256(final_dataset_path)
        )
        manifest_cache_matches = bool(
            manifest_path.exists()
            and str(cached.get("manifest_path", "")) == str(manifest_path.resolve())
            and str(cached.get("manifest_sha256", "")) == file_sha256(manifest_path)
        )
        robot_config_cache_matches = bool(
            robot_config_path.is_file()
            and str(cached.get("robot_config_path", "")) == str(robot_config_path)
            and str(cached.get("robot_config_sha256", "")) == file_sha256(robot_config_path)
        )
        family_coverage_cache_matches = bool(
            str(cached.get("formal_family_coverage_fingerprint", ""))
            == str(family_coverage["formal_family_coverage_fingerprint"])
            and bool(cached.get("formal_family_coverage_gate_pass", False))
            == bool(family_coverage["formal_family_coverage_gate_pass"])
            and all(
                path.is_file()
                and str(cached.get(f"{name}_path", "")) == str(path.resolve())
                and str(cached.get(f"{name}_sha256", "")) == file_sha256(path)
                for name, path in family_coverage_paths.items()
            )
        )
        if (
            phase_cache_is_compatible(
                cached,
                strategy_version=DATASET_STRATEGY_VERSION,
                task_fingerprint=task_fingerprint,
            )
            and dataset_cache_matches
            and manifest_cache_matches
            and robot_config_cache_matches
            and family_coverage_cache_matches
        ):
            return cached
    tube_dir = Path(args.out_dir) / "04_tube"
    tube_summary_path = tube_dir / "tube_radius_summary.csv"
    if not tube_summary_path.exists():
        raise FileNotFoundError("tube phase did not materialize the dataset inputs")
    tube_summary = pd.read_csv(tube_summary_path)
    passing = tube_summary[tube_summary["tube_gate_pass"].fillna(False).astype(bool)].copy()
    if passing.empty:
        report = {
            "strategy_version": DATASET_STRATEGY_VERSION,
            "task_fingerprint": task_fingerprint,
            **formal_expansion_protocol_report(args),
            **family_coverage_fields,
            "formal_dataset_gate_pass": False,
            "dataset_gate_pass": False,
            "trajectory_count": 0,
            "reason": "no_tube_passed",
        }
        write_json(report_path, report)
        return report

    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for _idx, source in passing.sort_values(["candidate_id", "radius_mm"]).iterrows():
        candidate_id = str(source["candidate_id"])
        family_id = str(source.get("family_id", candidate_id))
        radius_mm = float(source["radius_mm"])
        source_path = tube_dir / candidate_id / _radius_slug(radius_mm) / "tube_small.parquet"
        if not source_path.exists():
            raise FileNotFoundError(f"passing tube is missing its materialized dataset: {source_path}")
        tube = pd.read_parquet(source_path).copy()
        trajectory_id = f"{candidate_id}@{radius_mm:g}"
        tube["source_candidate_id"] = candidate_id
        tube["candidate_id"] = candidate_id
        tube["family_id"] = family_id
        tube["trajectory_id"] = trajectory_id
        tube["ellipse_id"] = trajectory_id
        tube["radius_mm"] = radius_mm
        tube["sample_id"] = [f"{trajectory_id}:{row_idx:06d}" for row_idx in range(len(tube))]
        angle_count = int(tube["angle_idx"].nunique())
        offset_count = int(tube["tube_offset_id"].nunique())
        expected_rows = int(args.final_points) * 25
        duplicate_samples = int(
            tube.duplicated(["angle_idx", "tube_offset_id"], keep=False).sum()
        )
        complete = bool(
            len(tube) == expected_rows
            and angle_count == int(args.final_points)
            and offset_count == 25
            and duplicate_samples == 0
        )
        manifest_rows.append(
            {
                "trajectory_id": trajectory_id,
                "candidate_id": candidate_id,
                "family_id": family_id,
                "radius_mm": radius_mm,
                "rows": int(len(tube)),
                "expected_rows": expected_rows,
                "angle_count": angle_count,
                "offset_count": offset_count,
                "duplicate_angle_offset_rows": duplicate_samples,
                "tube_success_ratio": float(tube["tube_success"].mean()),
                "trajectory_complete": complete,
                "source_path": str(source_path),
            }
        )
        frames.append(tube)
    diagnostic_union = pd.concat(frames, ignore_index=True, sort=False)
    manifest = pd.DataFrame(manifest_rows)
    required = {
        "trajectory_id",
        "sample_id",
        "angle_idx",
        "angle_rad",
        "tube_offset_id",
        "delta_n1_mm",
        "delta_n2_mm",
        "is_centerline",
        *atlas.TARGET_XYZ_COLS,
        *atlas.BETA_COLS,
    }
    missing_columns = sorted(required - set(diagnostic_union.columns))
    global_conflicts = branch_conflict_report(diagnostic_union, voxel_mm=2.0, threshold_deg=3.0)
    family_rows: list[dict[str, Any]] = []
    family_conflicts: dict[str, dict[str, Any]] = {}
    primary_radius_mm = float(getattr(args, "primary_radius_mm", 87.5))
    stretch_radius_mm = float(getattr(args, "stretch_radius_mm", 100.0))
    for family_id, family_data in diagnostic_union.groupby("family_id", sort=True):
        family_id = str(family_id)
        family_manifest = manifest[manifest["family_id"].astype(str).eq(family_id)]
        family_radii = np.asarray(sorted(family_data["radius_mm"].astype(float).unique()), dtype=float)
        conflicts = branch_conflict_report(family_data, voxel_mm=2.0, threshold_deg=3.0)
        family_conflicts[family_id] = conflicts
        trajectory_count = int(family_data["trajectory_id"].nunique())
        unique_radius_count = int(len(family_radii))
        complete = bool(family_manifest["trajectory_complete"].all())
        successful = bool(family_manifest["tube_success_ratio"].ge(0.99).all())
        unique_ids = bool(family_data["sample_id"].is_unique)
        eligible = bool(
            not missing_columns
            and unique_ids
            and complete
            and successful
            and conflicts["branch_conflict_gate_pass"]
            and trajectory_count >= 3
            and unique_radius_count >= 3
        )
        family_rows.append(
            {
                "family_id": family_id,
                "rows": int(len(family_data)),
                "trajectory_count": trajectory_count,
                "unique_radius_count": unique_radius_count,
                "min_radius_mm": float(np.min(family_radii)),
                "max_radius_mm": float(np.max(family_radii)),
                "has_primary_radius": bool(
                    np.any(np.isclose(family_radii, primary_radius_mm, atol=1.0e-8))
                ),
                "has_stretch_radius": bool(
                    np.any(np.isclose(family_radii, stretch_radius_mm, atol=1.0e-8))
                ),
                "unique_sample_ids": unique_ids,
                "all_trajectories_complete": complete,
                "all_source_tubes_successful": successful,
                "branch_conflict_gate_pass": bool(conflicts["branch_conflict_gate_pass"]),
                "conflict_voxels": int(conflicts["conflict_voxels"]),
                "max_voxel_beta_rms_deg": float(conflicts["max_voxel_beta_rms_deg"]),
                "dataset_family_eligible": eligible,
            }
        )
    family_selection = pd.DataFrame(family_rows).sort_values(
        [
            "dataset_family_eligible",
            "has_primary_radius",
            "has_stretch_radius",
            "max_radius_mm",
            "unique_radius_count",
            "trajectory_count",
            "family_id",
        ],
        ascending=[False, False, False, False, False, False, True],
        kind="stable",
    )
    family_selection.to_csv(out / "family_selection.csv", index=False)
    eligible = family_selection[family_selection["dataset_family_eligible"].astype(bool)]
    selected_family_id = str(eligible.iloc[0]["family_id"]) if not eligible.empty else None
    if selected_family_id is None:
        dataset = diagnostic_union.iloc[0:0].copy()
        conflicts = {
            "rows": 0,
            "voxel_mm": 2.0,
            "branch_conflict_threshold_deg": 3.0,
            "occupied_voxels": 0,
            "multi_sample_voxels": 0,
            "conflict_voxels": 0,
            "conflict_voxel_ratio": 0.0,
            "conflict_point_ratio": 0.0,
            "max_voxel_beta_rms_deg": 0.0,
            "branch_conflict_gate_pass": False,
            "reason": "no_eligible_fixed_family",
        }
    else:
        dataset = diagnostic_union[diagnostic_union["family_id"].astype(str).eq(selected_family_id)].copy()
        conflicts = family_conflicts[selected_family_id]
    manifest["selected_for_dataset"] = manifest["family_id"].astype(str).eq(selected_family_id)
    manifest.to_csv(manifest_path, index=False)
    unique_sample_ids = bool(len(dataset) and dataset["sample_id"].is_unique)
    selected_manifest = manifest[manifest["selected_for_dataset"]]
    all_complete = bool(len(selected_manifest) and selected_manifest["trajectory_complete"].all())
    all_success = bool(len(selected_manifest) and selected_manifest["tube_success_ratio"].ge(0.99).all())
    trajectory_count = int(dataset["trajectory_id"].nunique()) if len(dataset) else 0
    unique_radius_count = int(dataset["radius_mm"].nunique()) if len(dataset) else 0
    dataset_gate = bool(selected_family_id is not None)
    diagnostic_union.to_parquet(out / "dataset_attempt.parquet", index=False, compression="zstd")
    if dataset_gate:
        dataset.to_parquet(final_dataset_path, index=False, compression="zstd")
    dataset_sha256 = file_sha256(final_dataset_path) if dataset_gate else None
    manifest_sha256 = file_sha256(manifest_path)
    robot_config_sha256 = file_sha256(robot_config_path) if robot_config_path.is_file() else None
    family_coverage_binding_complete = bool(
        all(path.is_file() for path in family_coverage_paths.values())
        and all(family_coverage_fields.get(f"{name}_sha256") for name in family_coverage_paths)
    )
    artifact_binding_complete = bool(
        dataset_gate
        and dataset_sha256
        and manifest_sha256
        and robot_config_sha256
        and family_coverage_binding_complete
    )
    report = {
        "strategy_version": DATASET_STRATEGY_VERSION,
        "task_fingerprint": task_fingerprint,
        **formal_expansion_protocol_report(args),
        **family_coverage_fields,
        "formal_dataset_gate_pass": formal_dataset_claims_allowed(
            args,
            dataset_gate=dataset_gate,
            tube_report=tube_report,
            family_coverage_report=family_coverage,
            artifact_binding_complete=artifact_binding_complete,
        ),
        "dataset_gate_pass": dataset_gate,
        "rows": int(len(dataset)),
        "trajectory_count": trajectory_count,
        "unique_family_count": int(dataset["family_id"].nunique()) if len(dataset) else 0,
        "unique_radius_count": unique_radius_count,
        "radii_mm": sorted(float(value) for value in dataset["radius_mm"].unique()) if len(dataset) else [],
        "selected_family_id": selected_family_id,
        "all_tube_rows": int(len(diagnostic_union)),
        "all_tube_trajectory_count": int(diagnostic_union["trajectory_id"].nunique()),
        "all_tube_family_count": int(diagnostic_union["family_id"].nunique()),
        "missing_columns": missing_columns,
        "unique_sample_ids": unique_sample_ids,
        "all_trajectories_complete": all_complete,
        "all_source_tubes_successful": all_success,
        "minimum_trajectory_count": 3,
        "minimum_unique_radius_count": 3,
        "branch_conflict": conflicts,
        "global_branch_conflict": global_conflicts,
        "dataset_path": str(final_dataset_path.resolve()) if dataset_gate else None,
        "dataset_sha256": dataset_sha256,
        "dataset_bytes": int(final_dataset_path.stat().st_size) if dataset_gate else 0,
        "attempt_path": str(out / "dataset_attempt.parquet"),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "manifest_bytes": int(manifest_path.stat().st_size),
        "robot_config_path": str(robot_config_path),
        "robot_config_sha256": robot_config_sha256,
        "robot_config_bytes": int(robot_config_path.stat().st_size) if robot_config_path.is_file() else 0,
        "artifact_binding_complete": artifact_binding_complete,
        "family_selection_path": str(out / "family_selection.csv"),
    }
    write_json(report_path, report)
    return report


def build_expansion_radius_status(
    pointwise: pd.DataFrame,
    branch: pd.DataFrame,
    tube: pd.DataFrame,
    *,
    radii_mm: Iterable[float],
) -> pd.DataFrame:
    def passing_at(frame: pd.DataFrame, radius_mm: float, gate_col: str) -> set[str]:
        if frame.empty or gate_col not in frame.columns:
            return set()
        mask = np.isclose(frame["radius_mm"].to_numpy(dtype=float), float(radius_mm), atol=1.0e-8)
        return set(
            frame.loc[mask & frame[gate_col].fillna(False).astype(bool), "candidate_id"].astype(str).tolist()
        )

    rows: list[dict[str, Any]] = []
    for radius_mm in parse_float_csv(radii_mm):
        pointwise_families = passing_at(pointwise, radius_mm, "pointwise_gate_pass")
        branch_families = passing_at(branch, radius_mm, "branch_robustness_gate_pass")
        tube_families = passing_at(tube, radius_mm, "tube_gate_pass")
        after_branch = pointwise_families & branch_families
        passing = after_branch & tube_families
        if not pointwise_families:
            limiting = "pointwise_ik"
        elif not after_branch:
            limiting = "branch"
        elif not passing:
            limiting = "tube"
        else:
            limiting = "none"
        rows.append(
            {
                "radius_mm": float(radius_mm),
                "amp_z_mm": 1.5 * float(radius_mm),
                "pointwise_passing_family_count": int(len(pointwise_families)),
                "branch_passing_family_count": int(len(after_branch)),
                "tube_passing_family_count": int(len(passing)),
                "pointwise_passing_families": ",".join(sorted(pointwise_families)),
                "branch_passing_families": ",".join(sorted(after_branch)),
                "passing_families": ",".join(sorted(passing)),
                "trajectory_materialization_gate_pass": bool(passing),
                "limiting_factor": limiting,
            }
        )
    return pd.DataFrame(rows)


def phase_summary(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "06_summary"
    out.mkdir(parents=True, exist_ok=True)
    dataset_report_path = Path(args.out_dir) / "05_dataset" / "dataset_report.json"
    if not dataset_report_path.exists():
        phase_dataset(args)
    pointwise_path = Path(args.out_dir) / "02_pointwise" / "pointwise_radius_summary.csv"
    branch_path = Path(args.out_dir) / "03_branch" / "branch_radius_summary.csv"
    tube_path = Path(args.out_dir) / "04_tube" / "tube_radius_summary.csv"
    pointwise = pd.read_csv(pointwise_path) if pointwise_path.exists() else pd.DataFrame()
    branch = pd.read_csv(branch_path) if branch_path.exists() else pd.DataFrame()
    tube = pd.read_csv(tube_path) if tube_path.exists() else pd.DataFrame()
    anchors = [
        radius
        for radius in parse_float_csv(args.radius_anchors_mm)
        if radius <= float(args.stretch_radius_mm) + 1.0e-9
    ]
    status = build_expansion_radius_status(pointwise, branch, tube, radii_mm=anchors)
    status.to_csv(out / "radius_stage_status.csv", index=False)
    trajectory_rmax = connected_radius_max(
        status,
        gate_col="trajectory_materialization_gate_pass",
        anchor_mm=min(anchors),
    ) if anchors else None
    primary_row = _radius_row(status, float(args.primary_radius_mm))
    stretch_row = _radius_row(status, float(args.stretch_radius_mm))
    dataset_report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
    report = {
        **formal_expansion_protocol_report(args),
        "primary_radius_mm": float(args.primary_radius_mm),
        "stretch_radius_mm": float(args.stretch_radius_mm),
        "primary_trajectory_materialized": bool(
            primary_row is not None and primary_row["trajectory_materialization_gate_pass"]
        ),
        "stretch_trajectory_materialized": bool(
            stretch_row is not None and stretch_row["trajectory_materialization_gate_pass"]
        ),
        "connected_trajectory_rmax_mm": trajectory_rmax,
        "dataset_gate_pass": bool(dataset_report.get("dataset_gate_pass", False)),
        "formal_dataset_gate_pass": bool(dataset_report.get("formal_dataset_gate_pass", False)),
        "model_training_executed": False,
        "primary_strict_support_claimed": False,
        "stretch_strict_support_claimed": False,
        "radius_status_path": str(out / "radius_stage_status.csv"),
        "dataset_report": dataset_report,
    }
    write_json(out / "expansion_summary.json", report)
    lines = [
        "# True Ellipse V5 family expansion summary",
        "",
        f"- Primary radius: `{float(args.primary_radius_mm):g} mm`.",
        f"- Stretch radius: `{float(args.stretch_radius_mm):g} mm`.",
        f"- Connected materialized trajectory radius: `{trajectory_rmax if trajectory_rmax is not None else 'none'} mm`.",
        f"- Primary robust tube materialized: `{report['primary_trajectory_materialized']}`.",
        f"- Stretch robust tube materialized: `{report['stretch_trajectory_materialized']}`.",
        f"- Multi-trajectory dataset gate: `{report['dataset_gate_pass']}`.",
        f"- Formal expansion/dataset gate: `{report['formal_expansion_protocol_gate_pass']}/{report['formal_dataset_gate_pass']}`.",
        "- This expansion stage does not claim strict model support; that claim is produced only by the V5 training/generalization runner.",
        "",
        "## Radius stage status",
        "",
        dataframe_to_markdown(status),
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _radius_row(table: pd.DataFrame, radius_mm: float) -> pd.Series | None:
    mask = np.isclose(table["radius_mm"].to_numpy(dtype=float), float(radius_mm), atol=1.0e-8)
    if not np.any(mask):
        return None
    return table.loc[mask].iloc[0]


def build_goal_report(
    radius_status: pd.DataFrame,
    *,
    primary_radius_mm: float,
    stretch_radius_mm: float,
    anchor_mm: float = 75.0,
) -> dict[str, Any]:
    strict_rmax = connected_radius_max(radius_status, gate_col="strict_gate_pass", anchor_mm=float(anchor_mm))
    generalized_rmax = connected_radius_max(
        radius_status,
        gate_col="trajectory_generalization_gate_pass",
        anchor_mm=float(anchor_mm),
    )
    primary = _radius_row(radius_status, float(primary_radius_mm))
    stretch = _radius_row(radius_status, float(stretch_radius_mm))
    primary_pass = bool(
        primary is not None
        and bool(primary["strict_gate_pass"])
        and bool(primary["trajectory_generalization_gate_pass"])
    )
    stretch_pass = bool(stretch is not None and bool(stretch["strict_gate_pass"]))
    stretch_generalization = bool(
        stretch is not None
        and bool(stretch["strict_gate_pass"])
        and bool(stretch["trajectory_generalization_gate_pass"])
    )
    return {
        "primary_radius_mm": float(primary_radius_mm),
        "stretch_radius_mm": float(stretch_radius_mm),
        "primary_goal_pass": primary_pass,
        "stretch_radius_pass": stretch_pass,
        "stretch_generalization_pass": stretch_generalization,
        "strict_supported_rmax_mm": strict_rmax,
        "trajectory_generalized_rmax_mm": generalized_rmax,
        "stretch_limiting_factor": None if stretch is None else str(stretch.get("limiting_factor", "unknown")),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand the true-ellipse canonical family to 87.5mm and strictly challenge 100mm.")
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2)
    parser.add_argument("--v3-dir", type=Path, default=DEFAULT_V3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phases", default="all")
    parser.add_argument("--primary-radius-mm", type=float, default=87.5)
    parser.add_argument("--stretch-radius-mm", type=float, default=100.0)
    parser.add_argument("--radius-anchors-mm", default="75,80,82.5,85,87.5,90,92.5,95,97.5,100")
    parser.add_argument("--family-sobol-samples", type=int, default=2048)
    parser.add_argument("--pointwise-seed-budgets", default="16,32,64")
    parser.add_argument("--max-pointwise-families", type=int, default=5)
    parser.add_argument("--max-branch-families", type=int, default=5)
    parser.add_argument("--tube-offsets-mm", default="-5,-2.5,0,2.5,5")
    parser.add_argument("--coarse-points", type=int, default=72)
    parser.add_argument("--final-points", type=int, default=360)
    parser.add_argument("--max-ik-nfev", type=int, default=200)
    parser.add_argument("--max-opt-nfev", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--worker-task", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    phases = parse_phases(args.phases)
    handlers = {
        "audit": phase_audit,
        "search": phase_search,
        "pointwise": phase_pointwise,
        "branch": phase_branch,
        "tube": phase_tube,
        "dataset": phase_dataset,
        "summary": phase_summary,
    }
    results: dict[str, Any] = {}
    for phase in phases:
        if phase not in handlers:
            raise NotImplementedError(f"V5 phase is not implemented yet: {phase}")
        results[phase] = handlers[phase](args)
    payload = {
        "mode": "true_ellipse_family_expansion_v5",
        "phases": phases,
        "out_dir": str(args.out_dir),
        **formal_expansion_protocol_report(args),
        "results": results,
    }
    write_json(Path(args.out_dir) / "run_report.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    if args.worker_task is not None:
        run_worker_task(Path(args.worker_task))
        return 0
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
