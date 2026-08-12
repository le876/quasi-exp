"""Deterministic, fail-closed contracts for subprocess audit shards.

The numerical worker implementation is intentionally outside this module.  It
owns only the byte-independent scientific contract: a schedule is assigned to
one stable shard, every schedule must produce the complete direction/repeat
cartesian product, and completed shards are reusable only under the same
source, configuration, phase, registry, and execution payload.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


_EXECUTION_KEY = ("schedule_id", "direction", "repeat_index")
_DIRECTIONS = ("forward", "reverse")


def add_schedule_waypoint_estimates(
    schedules: pd.DataFrame,
    task_nodes: pd.DataFrame,
    *,
    maximum_step_mm: float,
) -> pd.DataFrame:
    """Return schedules with deterministic geometric waypoint-count priors."""

    if maximum_step_mm <= 0.0:
        raise ValueError("maximum_step_mm must be positive")
    required = {"task_node_id", "x_m", "y_m", "z_m"}
    missing = required - set(task_nodes)
    if missing:
        raise ValueError(f"task nodes lack waypoint-estimation columns: {sorted(missing)}")
    xyz_by_node = {
        int(row.task_node_id): np.asarray([row.x_m, row.y_m, row.z_m], dtype=float)
        for row in task_nodes.itertuples(index=False)
    }

    def estimate(path: Any) -> int:
        if not isinstance(path, (list, tuple, np.ndarray)) or len(path) < 2:
            return 1
        total = 0
        for left, right in zip(path[:-1], path[1:], strict=True):
            if int(left) not in xyz_by_node or int(right) not in xyz_by_node:
                raise ValueError(f"schedule path references unknown task node: {left}->{right}")
            distance_mm = float(
                np.linalg.norm(xyz_by_node[int(right)] - xyz_by_node[int(left)]) * 1000.0
            )
            total += max(1, int(math.ceil(distance_mm / float(maximum_step_mm))))
        return max(1, total)

    result = schedules.copy()
    paths = result.get("path_node_ids", pd.Series([()] * len(result), index=result.index))
    result["estimated_waypoint_count"] = paths.map(estimate).astype(int)
    return result


@dataclass(frozen=True)
class AuditCostModel:
    """Deterministic static cost prior for assigning immutable schedules."""

    direction_count: int = 2
    repeats_per_direction: int = 3
    audit_kind_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "edge": 1.0,
            "root_path": 1.15,
            "fundamental_cycle": 1.25,
            "multipath_tree": 1.15,
            "multipath_chord": 1.0,
        }
    )

    def estimate(self, row: Mapping[str, Any]) -> tuple[int, int, float]:
        path = row.get("path_node_ids", ())
        path_edge_count = max(1, len(path) - 1) if isinstance(path, (list, tuple, np.ndarray)) else 1
        waypoint_count = int(row.get("estimated_waypoint_count", path_edge_count))
        waypoint_count = max(path_edge_count, waypoint_count)
        kind_weight = float(self.audit_kind_weights.get(str(row.get("audit_kind", "")), 1.0))
        cost = (
            int(self.direction_count)
            * int(self.repeats_per_direction)
            * waypoint_count
            * kind_weight
        )
        return path_edge_count, waypoint_count, float(cost)


def _strict(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _frame_digest(frame: pd.DataFrame, *, sort_columns: tuple[str, ...]) -> str:
    if frame.empty:
        records: list[dict[str, Any]] = []
    else:
        missing = set(sort_columns) - set(frame)
        if missing:
            raise ValueError(f"digest frame is missing sort columns: {sorted(missing)}")
        ordered = frame.sort_values(list(sort_columns), kind="stable").reset_index(drop=True)
        records = [_strict(record) for record in ordered.to_dict(orient="records")]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_audit_shard_registry(
    schedules: pd.DataFrame,
    *,
    shard_count: int,
    repeats_per_direction: int,
    assignment_strategy: str = "hash",
    cost_model: AuditCostModel | None = None,
) -> pd.DataFrame:
    """Assign every registered schedule to exactly one stable subprocess shard."""

    if shard_count < 1 or repeats_per_direction < 1:
        raise ValueError("shard and repeat counts must be positive")
    if "schedule_id" not in schedules:
        raise ValueError("audit schedules require schedule_id")
    if schedules["schedule_id"].astype(str).duplicated().any():
        raise ValueError("audit schedule_id must be unique before sharding")
    registry = schedules.copy()
    registry["schedule_id"] = registry["schedule_id"].astype(str)
    strategy = str(assignment_strategy).strip().lower()
    if strategy not in {"hash", "cost_balanced_lpt"}:
        raise ValueError(f"unknown audit shard assignment strategy: {assignment_strategy}")
    if strategy == "hash":
        registry["estimated_path_edge_count"] = registry.get(
            "path_node_ids", pd.Series([()] * len(registry), index=registry.index)
        ).map(lambda path: max(1, len(path) - 1) if isinstance(path, (list, tuple, np.ndarray)) else 1)
        registry["estimated_waypoint_count"] = registry["estimated_path_edge_count"]
        registry["estimated_cost"] = (
            registry["estimated_waypoint_count"].astype(float)
            * 2.0
            * float(repeats_per_direction)
        )
        registry["shard_id"] = registry["schedule_id"].map(
            lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
            % int(shard_count)
        )
    else:
        model = cost_model or AuditCostModel(repeats_per_direction=int(repeats_per_direction))
        estimates = [model.estimate(row) for row in registry.to_dict(orient="records")]
        registry["estimated_path_edge_count"] = [value[0] for value in estimates]
        registry["estimated_waypoint_count"] = [value[1] for value in estimates]
        registry["estimated_cost"] = [value[2] for value in estimates]
        loads = [0.0] * int(shard_count)
        assignment: dict[str, int] = {}
        ranked = sorted(
            registry.to_dict(orient="records"),
            key=lambda row: (-float(row["estimated_cost"]), str(row["schedule_id"])),
        )
        for row in ranked:
            shard_id = min(range(int(shard_count)), key=lambda value: (loads[value], value))
            schedule_id = str(row["schedule_id"])
            assignment[schedule_id] = shard_id
            loads[shard_id] += float(row["estimated_cost"])
        registry["shard_id"] = registry["schedule_id"].map(assignment).astype(int)
    registry["assignment_strategy"] = strategy
    registry["registered_shard_count"] = int(shard_count)
    registry["expected_execution_count"] = 2 * int(repeats_per_direction)
    return registry.sort_values("schedule_id", kind="stable").reset_index(drop=True)


def _coverage_checks(
    registry: pd.DataFrame,
    executions: pd.DataFrame,
    *,
    repeats_per_direction: int,
) -> dict[str, Any]:
    schedule_ids = tuple(registry.get("schedule_id", pd.Series(dtype=str)).astype(str))
    expected = {
        (schedule_id, direction, repeat_index)
        for schedule_id in schedule_ids
        for direction in _DIRECTIONS
        for repeat_index in range(int(repeats_per_direction))
    }
    missing_columns = set(_EXECUTION_KEY) - set(executions)
    if missing_columns:
        observed: list[tuple[str, str, int]] = []
        duplicate_count = 0
    else:
        observed = [
            (str(row.schedule_id), str(row.direction), int(row.repeat_index))
            for row in executions.loc[:, list(_EXECUTION_KEY)].itertuples(index=False)
        ]
        duplicate_count = len(observed) - len(set(observed))
    observed_set = set(observed)
    missing = sorted(expected - observed_set)
    unexpected = sorted(observed_set - expected)
    checks = {
        "execution_key_columns": not missing_columns,
        "exact_execution_keys": not missing and not unexpected and duplicate_count == 0,
        "exact_execution_count": len(observed) == len(expected),
    }
    return {
        "gate_pass": bool(all(checks.values())),
        "checks": checks,
        "expected_execution_count": len(expected),
        "observed_execution_count": len(observed),
        "duplicate_execution_key_count": duplicate_count,
        "missing_execution_key_count": len(missing),
        "unexpected_execution_key_count": len(unexpected),
        "missing_execution_keys": missing[:32],
        "unexpected_execution_keys": unexpected[:32],
    }


def build_shard_completion_report(
    registry: pd.DataFrame,
    executions: pd.DataFrame,
    *,
    source_sha: str,
    config_sha256: str,
    phase_id: str,
    shard_id: int,
    repeats_per_direction: int,
    input_sha256: str = "",
) -> dict[str, Any]:
    coverage = _coverage_checks(
        registry, executions, repeats_per_direction=repeats_per_direction
    )
    return {
        "schema_version": 1,
        "gate_pass": bool(coverage["gate_pass"]),
        "source_sha": str(source_sha),
        "config_sha256": str(config_sha256),
        "input_sha256": str(input_sha256),
        "phase_id": str(phase_id),
        "shard_id": int(shard_id),
        "schedule_count": len(registry),
        "execution_count": len(executions),
        "registry_sha256": _frame_digest(registry, sort_columns=("schedule_id",)),
        "executions_sha256": _frame_digest(
            executions, sort_columns=_EXECUTION_KEY
        ),
        "coverage": coverage,
    }


def validate_shard_completion(
    report: Mapping[str, Any],
    registry: pd.DataFrame,
    executions: pd.DataFrame,
    *,
    source_sha: str,
    config_sha256: str,
    phase_id: str,
    shard_id: int,
    repeats_per_direction: int,
    input_sha256: str = "",
) -> dict[str, Any]:
    coverage = _coverage_checks(
        registry, executions, repeats_per_direction=repeats_per_direction
    )
    registry_sha = _frame_digest(registry, sort_columns=("schedule_id",))
    execution_sha = _frame_digest(executions, sort_columns=_EXECUTION_KEY)
    checks = {
        "report_gate": bool(report.get("gate_pass", False)),
        "source_sha": str(report.get("source_sha", "")) == str(source_sha),
        "config_sha256": str(report.get("config_sha256", "")) == str(config_sha256),
        "input_sha256": str(report.get("input_sha256", "")) == str(input_sha256),
        "phase_id": str(report.get("phase_id", "")) == str(phase_id),
        "shard_id": int(report.get("shard_id", -1)) == int(shard_id),
        "schedule_count": int(report.get("schedule_count", -1)) == len(registry),
        "execution_count": int(report.get("execution_count", -1)) == len(executions),
        "registry_sha256": str(report.get("registry_sha256", "")) == registry_sha,
        "executions_sha256": str(report.get("executions_sha256", "")) == execution_sha,
        "exact_execution_keys": bool(coverage["checks"]["exact_execution_keys"]),
        "exact_execution_count": bool(coverage["checks"]["exact_execution_count"]),
    }
    return {
        "gate_pass": bool(all(checks.values())),
        "checks": checks,
        "coverage": coverage,
        "registry_sha256": registry_sha,
        "executions_sha256": execution_sha,
    }


def merge_validated_audit_shards(
    registry: pd.DataFrame,
    shard_executions: Mapping[int, pd.DataFrame],
    *,
    repeats_per_direction: int,
) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame(columns=list(_EXECUTION_KEY))
    registered_count = int(registry["registered_shard_count"].iloc[0])
    if not registry["registered_shard_count"].eq(registered_count).all():
        raise ValueError("registry contains inconsistent shard counts")
    expected_shards = set(range(registered_count))
    actual_shards = set(map(int, shard_executions))
    missing_shards = sorted(expected_shards - actual_shards)
    unexpected_shards = sorted(actual_shards - expected_shards)
    if missing_shards:
        raise ValueError(f"missing shard outputs: {missing_shards}")
    if unexpected_shards:
        raise ValueError(f"unexpected shard outputs: {unexpected_shards}")
    frames: list[pd.DataFrame] = []
    for shard_id in sorted(expected_shards):
        shard_registry = registry[registry["shard_id"].eq(shard_id)]
        frame = shard_executions[shard_id]
        coverage = _coverage_checks(
            shard_registry, frame, repeats_per_direction=repeats_per_direction
        )
        if not coverage["gate_pass"]:
            raise ValueError(
                f"shard {shard_id} does not have exact execution coverage: {coverage}"
            )
        frames.append(frame)
    nonempty = [frame for frame in frames if len(frame)]
    if nonempty:
        merged = pd.concat(nonempty, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=list(_EXECUTION_KEY))
    coverage = _coverage_checks(
        registry, merged, repeats_per_direction=repeats_per_direction
    )
    if not coverage["gate_pass"]:
        raise ValueError(f"aggregate does not have exact execution coverage: {coverage}")
    return merged.sort_values(list(_EXECUTION_KEY), kind="stable").reset_index(drop=True)
