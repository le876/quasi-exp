#!/usr/bin/env python3
"""IPC-safe numerical worker for retry14 shell-mesh continuation.

The worker loads the robot environment once, consumes one JSON batch, and emits
exactly one JSON object on stdout.  It never writes mesh or scientific artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (SOURCE_ROOT / "src", SOURCE_ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import yaml

from run_trajectory_canonical_teacher_v10 import load_environment
from quasi_exp.teacher.canonical_atlas import AtlasCandidate, AtlasTaskNode
from quasi_exp.teacher.optimized_continuation import (
    make_iterative_weighted_dls_continuation,
    make_optimized_predictor_corrector_continuation,
)
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.retry10 import normalized_weighted_beta_deg, raw_beta_max_deg


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _environment(config: Mapping[str, Any]) -> Any:
    robot_path = SOURCE_ROOT / str(config["sources"]["robot_config"])
    project_root = SOURCE_ROOT.resolve()
    if project_root.parent.name == ".worktrees":
        project_root = project_root.parent.parent
    return optimized_forward(load_environment(project_root, robot_path))


def _candidate(task: Mapping[str, Any]) -> AtlasCandidate:
    beta = np.asarray(task["source_beta"], dtype=float)
    return AtlasCandidate(
        node_id=int(task.get("source_node_id", 0)),
        candidate_id=str(task["source_vertex_id"]),
        beta_rad=beta,
        residual_mm=float(task.get("source_fk_residual_mm", 0.0)),
        min_margin_deg=float(task.get("source_minimum_margin_deg", 1.0)),
        normalized_min_margin=1.0,
        posture_cost=float(np.linalg.norm(beta)),
        condition_number=float(task.get("source_condition_number", 1.0)),
        quality="Gold",
        solver_success=True,
        actual_bounds=True,
    )


def _constrained_solve(
    environment: Any,
    source_beta: np.ndarray,
    target_xyz: np.ndarray,
    seam_class: str,
) -> tuple[np.ndarray, bool, float, str]:
    from scipy.optimize import least_squares

    if seam_class == "y_seam":
        free = np.asarray([1, 3, 5], dtype=int)
    elif seam_class == "z_seam":
        free = np.asarray([0, 2, 4], dtype=int)
    else:
        raise ValueError(f"unsupported constrained seam {seam_class!r}")
    bounds = np.asarray(environment.bounds, dtype=float)
    initial = np.asarray(source_beta, dtype=float).copy()
    fixed = np.ones(6, dtype=bool)
    fixed[free] = False
    initial[fixed] = 0.0

    def residual(free_beta: np.ndarray) -> np.ndarray:
        beta = initial.copy()
        beta[free] = free_beta
        return np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)[0] - target_xyz

    result = least_squares(
        residual,
        initial[free],
        bounds=(bounds[free, 0], bounds[free, 1]),
        max_nfev=400,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    beta = initial.copy()
    beta[free] = result.x
    residual_mm = float(np.linalg.norm(residual(result.x)) * 1000.0)
    success = bool(result.success and residual_mm <= 8.0 and np.isfinite(beta).all())
    return beta, success, residual_mm, f"constrained:{result.status}"


def _solve_direct(
    environment: Any,
    task: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    source_beta = np.asarray(task["source_beta"], dtype=float)
    source_xyz = np.asarray(task["source_xyz"], dtype=float)
    target_xyz = np.asarray(task["target_xyz"], dtype=float)
    seam_class = str(task.get("seam_class", "interior"))
    optimized = make_optimized_predictor_corrector_continuation(
        environment,
        damping=float(config["solver"]["damping"]),
        max_corrector_iterations=int(config["solver"]["maximum_corrector_iterations"]),
        residual_tolerance_mm=float(config["solver"]["fk_residual_maximum_mm"]),
    )
    iterative = make_iterative_weighted_dls_continuation(
        environment,
        damping=float(config["solver"]["damping"]),
        max_corrector_iterations=int(config["solver"]["maximum_corrector_iterations"]),
        residual_tolerance_mm=float(config["solver"]["fk_residual_maximum_mm"]),
    )
    method = "optimized"
    status = ""
    minimum_margin = math.nan
    if seam_class in {"y_seam", "z_seam"}:
        beta, success, residual_mm, status = _constrained_solve(
            environment, source_beta, target_xyz, seam_class
        )
        actual_bounds = success
        reverse_method = None
        method = f"constrained_{seam_class}"
    else:
        source = _candidate(task)
        target = AtlasTaskNode(
            int(task.get("target_node_id", 1)), target_xyz, ()
        )
        outcome = optimized(source, target)
        reverse_method = optimized
        if not outcome.success or not outcome.actual_bounds:
            fallback = iterative(source, target)
            if fallback.success or float(fallback.residual_mm) < float(outcome.residual_mm):
                outcome = fallback
                reverse_method = iterative
                method = "iterative"
        beta = np.asarray(outcome.beta_rad, dtype=float)
        success = bool(outcome.success)
        actual_bounds = bool(outcome.actual_bounds)
        residual_mm = float(outcome.residual_mm)
        minimum_margin = float(outcome.minimum_margin_deg or math.nan)
        status = str(outcome.status)

    reverse_success = False
    reverse_weighted_gap_deg = math.inf
    reverse_raw_gap_deg = math.inf
    if success and actual_bounds and np.isfinite(beta).all():
        endpoint = AtlasCandidate(
            node_id=int(task.get("target_node_id", 1)),
            candidate_id=f"retry14_endpoint:{task['task_id']}",
            beta_rad=beta,
            residual_mm=residual_mm,
            min_margin_deg=minimum_margin if np.isfinite(minimum_margin) else 1.0,
            normalized_min_margin=1.0,
            posture_cost=float(np.linalg.norm(beta)),
            condition_number=1.0,
            quality="Gold",
            solver_success=True,
            actual_bounds=True,
        )
        if seam_class in {"y_seam", "z_seam"}:
            reverse_beta, reverse_success, _, _ = _constrained_solve(
                environment, beta, source_xyz, seam_class
            )
            if reverse_success:
                reverse_weighted_gap_deg = normalized_weighted_beta_deg(
                    reverse_beta, source_beta
                )
                reverse_raw_gap_deg = raw_beta_max_deg(reverse_beta, source_beta)
        else:
            reverse_target = AtlasTaskNode(
                int(task.get("source_node_id", 0)), source_xyz, ()
            )
            reverse = reverse_method(endpoint, reverse_target)
            reverse_success = bool(reverse.success and reverse.actual_bounds)
            if reverse_success:
                reverse_weighted_gap_deg = normalized_weighted_beta_deg(
                    np.asarray(reverse.beta_rad, dtype=float), source_beta
                )
                reverse_raw_gap_deg = raw_beta_max_deg(
                    np.asarray(reverse.beta_rad, dtype=float), source_beta
                )
    bounds = np.asarray(environment.bounds, dtype=float)
    finite = bool(np.isfinite(beta).all() and np.isfinite(residual_mm))
    bounds_pass = bool(
        finite
        and np.all(beta >= bounds[:, 0] - 1.0e-12)
        and np.all(beta <= bounds[:, 1] + 1.0e-12)
    )
    admitted = bool(
        success
        and actual_bounds
        and finite
        and bounds_pass
        and residual_mm <= float(config["solver"]["fk_residual_maximum_mm"])
    )
    return {
        "task_id": str(task["task_id"]),
        "mesh_vertex_id": str(task["mesh_vertex_id"]),
        "source_vertex_id": str(task["source_vertex_id"]),
        "logical_edge_id": str(task.get("logical_edge_id", "")),
        "solver_segment_id": str(task.get("solver_segment_id", "")),
        "mesh_edge_type": str(task.get("mesh_edge_type", "")),
        "lineage_id": str(task.get("lineage_id", "")),
        "step_size_mm": float(task["step_size_mm"]),
        "solver_method": method,
        "solver_status": status,
        "solver_success": bool(success),
        "actual_bounds": bool(actual_bounds and bounds_pass),
        "finite": finite,
        "fk_residual_mm": residual_mm,
        "minimum_margin_deg": minimum_margin,
        "reverse_success": reverse_success,
        "reverse_weighted_gap_deg": reverse_weighted_gap_deg,
        "reverse_raw_gap_deg": reverse_raw_gap_deg,
        "source_to_endpoint_weighted_gap_deg": normalized_weighted_beta_deg(beta, source_beta),
        "source_to_endpoint_raw_gap_deg": raw_beta_max_deg(beta, source_beta),
        "admitted": admitted,
        "beta": beta.tolist(),
        "beta_sha256": hashlib.sha256(beta.astype("<f8").tobytes()).hexdigest(),
    }


def _solve_one(
    environment: Any,
    task: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve one registered segment, genuinely refining failed corridors.

    A 2.5/1.25 mm tier is represented by an explicit chain of short numerical
    solves.  Every subsegment must be admitted and reverse-certified; the
    caller still receives one immutable result for the registered segment.
    """

    tier_text = str(task.get("solver_tier", "5.0mm")).removesuffix("mm")
    maximum_step_mm = float(tier_text)
    source_xyz = np.asarray(task["source_xyz"], dtype=float)
    target_xyz = np.asarray(task["target_xyz"], dtype=float)
    length_mm = float(np.linalg.norm(target_xyz - source_xyz) * 1000.0)
    count = max(1, int(np.ceil(length_mm / maximum_step_mm)))
    if count == 1:
        result = _solve_direct(environment, task, config)
        result["refinement_subsegment_count"] = 1
        result["maximum_refinement_step_mm"] = length_mm
        return result
    original_beta = np.asarray(task["source_beta"], dtype=float)
    current_beta = original_beta.copy()
    current_xyz = source_xyz.copy()
    results: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        waypoint = source_xyz + (target_xyz - source_xyz) * index / count
        subtask = dict(task)
        subtask.update(
            {
                "task_id": f"{task['task_id']}:sub:{index}:{count}",
                "source_beta": current_beta.tolist(),
                "source_xyz": current_xyz.tolist(),
                "target_xyz": waypoint.tolist(),
                "step_size_mm": float(np.linalg.norm(waypoint - current_xyz) * 1000.0),
                "source_vertex_id": f"{task['source_vertex_id']}:sub:{index - 1}",
            }
        )
        result = _solve_direct(environment, subtask, config)
        results.append(result)
        if not bool(result.get("admitted", False)) or not bool(result.get("reverse_success", False)):
            failed = dict(result)
            failed.update(
                {
                    "task_id": str(task["task_id"]),
                    "mesh_vertex_id": str(task["mesh_vertex_id"]),
                    "source_vertex_id": str(task["source_vertex_id"]),
                    "logical_edge_id": str(task.get("logical_edge_id", "")),
                    "solver_segment_id": str(task.get("solver_segment_id", "")),
                    "refinement_subsegment_count": count,
                    "refinement_failed_subsegment": index,
                    "maximum_refinement_step_mm": length_mm / count,
                    "admitted": False,
                }
            )
            return failed
        current_beta = np.asarray(result["beta"], dtype=float)
        current_xyz = waypoint
    final = dict(results[-1])
    final.update(
        {
            "task_id": str(task["task_id"]),
            "mesh_vertex_id": str(task["mesh_vertex_id"]),
            "source_vertex_id": str(task["source_vertex_id"]),
            "logical_edge_id": str(task.get("logical_edge_id", "")),
            "solver_segment_id": str(task.get("solver_segment_id", "")),
            "step_size_mm": length_mm,
            "source_to_endpoint_weighted_gap_deg": normalized_weighted_beta_deg(current_beta, original_beta),
            "source_to_endpoint_raw_gap_deg": raw_beta_max_deg(current_beta, original_beta),
            "reverse_success": all(bool(value.get("reverse_success", False)) for value in results),
            "reverse_weighted_gap_deg": max(float(value.get("reverse_weighted_gap_deg", math.inf)) for value in results),
            "reverse_raw_gap_deg": max(float(value.get("reverse_raw_gap_deg", math.inf)) for value in results),
            "refinement_subsegment_count": count,
            "maximum_refinement_step_mm": length_mm / count,
            "admitted": all(bool(value.get("admitted", False)) for value in results),
            "beta": current_beta.tolist(),
            "beta_sha256": hashlib.sha256(current_beta.astype("<f8").tobytes()).hexdigest(),
        }
    )
    return final


def run_batch(config_path: Path, task_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    tasks = list(payload.get("tasks", []))
    environment = _environment(config)
    results: list[dict[str, Any]] = []
    for task in tasks:
        try:
            results.append(_solve_one(environment, task, config))
        except Exception as exc:  # fail task, not the whole evidence batch
            results.append(
                {
                    "task_id": str(task.get("task_id", "unknown")),
                    "mesh_vertex_id": str(task.get("mesh_vertex_id", "unknown")),
                    "source_vertex_id": str(task.get("source_vertex_id", "unknown")),
                    "admitted": False,
                    "solver_success": False,
                    "actual_bounds": False,
                    "finite": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {
        "batch_id": str(payload.get("batch_id", task_path.stem)),
        "task_count": len(tasks),
        "results": results,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    value = run_batch(args.config, args.task_file)
    print(json.dumps(value, default=_json_default, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
