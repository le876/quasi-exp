from __future__ import annotations

import pandas as pd
import pytest

from quasi_exp.teacher.audit_shards import (
    AuditCostModel,
    add_schedule_waypoint_estimates,
    build_audit_shard_registry,
    build_shard_completion_report,
    merge_validated_audit_shards,
    validate_shard_completion,
)


def _schedules() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "schedule_id": f"schedule_{index:02d}",
                "unique_entity_id": f"chart_000:edge:{index}:{index + 1}",
                "chart_id": "chart_000",
                "audit_kind": "edge",
                "path_node_ids": [index, index + 1],
                "primary_usage": True,
            }
            for index in range(12)
        ]
    )


def _executions(registry: pd.DataFrame, repeats: int = 3) -> pd.DataFrame:
    rows = []
    for schedule_id in registry["schedule_id"].astype(str):
        for direction in ("forward", "reverse"):
            for repeat_index in range(repeats):
                rows.append(
                    {
                        "schedule_id": schedule_id,
                        "direction": direction,
                        "repeat_index": repeat_index,
                        "solver_success": True,
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_registry_assignment_is_stable_and_covers_every_schedule_once() -> None:
    schedules = _schedules()
    first = build_audit_shard_registry(
        schedules, shard_count=4, repeats_per_direction=3
    )
    second = build_audit_shard_registry(
        schedules.sample(frac=1.0, random_state=71),
        shard_count=4,
        repeats_per_direction=3,
    )

    assert first["schedule_id"].is_unique
    assert set(first["schedule_id"]) == set(schedules["schedule_id"])
    assert first.set_index("schedule_id")["shard_id"].to_dict() == second.set_index(
        "schedule_id"
    )["shard_id"].to_dict()
    assert set(first["expected_execution_count"]) == {6}
    assert first["shard_id"].between(0, 3).all()


def test_cost_balanced_registry_is_deterministic_and_preserves_exact_set() -> None:
    schedules = _schedules()
    schedules.loc[:, "audit_kind"] = [
        "root_path" if index % 3 == 0 else "edge" for index in range(len(schedules))
    ]
    schedules["path_node_ids"] = pd.Series(
        [list(range(index + 2)) for index in range(len(schedules))], dtype=object
    )
    model = AuditCostModel(repeats_per_direction=3)
    first = build_audit_shard_registry(
        schedules,
        shard_count=7,
        repeats_per_direction=3,
        assignment_strategy="cost_balanced_lpt",
        cost_model=model,
    )
    second = build_audit_shard_registry(
        schedules.sample(frac=1.0, random_state=12),
        shard_count=7,
        repeats_per_direction=3,
        assignment_strategy="cost_balanced_lpt",
        cost_model=model,
    )
    assert first.set_index("schedule_id")["shard_id"].to_dict() == second.set_index(
        "schedule_id"
    )["shard_id"].to_dict()
    assert set(first["schedule_id"]) == set(schedules["schedule_id"])
    assert set(first["assignment_strategy"]) == {"cost_balanced_lpt"}
    assert (first["estimated_cost"] > 0).all()


def test_more_logical_shards_do_not_change_expected_execution_set() -> None:
    schedules = _schedules()
    execution_sets = []
    for shard_count in (12, 48, 96):
        registry = build_audit_shard_registry(
            schedules,
            shard_count=shard_count,
            repeats_per_direction=3,
            assignment_strategy="cost_balanced_lpt",
        )
        execution_sets.append(
            set(
                _executions(registry)[
                    ["schedule_id", "direction", "repeat_index"]
                ].itertuples(index=False, name=None)
            )
        )
    assert execution_sets[0] == execution_sets[1] == execution_sets[2]


def test_geometric_waypoint_estimate_counts_long_edges() -> None:
    schedules = pd.DataFrame.from_records(
        [{"schedule_id": "long", "path_node_ids": [0, 1, 2]}]
    )
    nodes = pd.DataFrame.from_records(
        [
            {"task_node_id": 0, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
            {"task_node_id": 1, "x_m": 0.012, "y_m": 0.0, "z_m": 0.0},
            {"task_node_id": 2, "x_m": 0.012, "y_m": 0.009, "z_m": 0.0},
        ]
    )
    estimated = add_schedule_waypoint_estimates(
        schedules, nodes, maximum_step_mm=5.0
    )
    assert int(estimated.loc[0, "estimated_waypoint_count"]) == 5


def test_shard_completion_requires_exact_execution_keys_and_matching_hashes() -> None:
    registry = build_audit_shard_registry(
        _schedules(), shard_count=3, repeats_per_direction=3
    )
    shard = registry[registry["shard_id"].eq(1)].copy()
    executions = _executions(shard)
    report = build_shard_completion_report(
        shard,
        executions,
        source_sha="source-a",
        config_sha256="config-a",
        phase_id="chart_initial",
        shard_id=1,
        repeats_per_direction=3,
    )

    valid = validate_shard_completion(
        report,
        shard,
        executions,
        source_sha="source-a",
        config_sha256="config-a",
        phase_id="chart_initial",
        shard_id=1,
        repeats_per_direction=3,
    )
    assert valid["gate_pass"] is True

    tampered = executions.iloc[:-1].copy()
    invalid = validate_shard_completion(
        report,
        shard,
        tampered,
        source_sha="source-a",
        config_sha256="config-a",
        phase_id="chart_initial",
        shard_id=1,
        repeats_per_direction=3,
    )
    assert invalid["gate_pass"] is False
    assert invalid["checks"]["exact_execution_keys"] is False

    wrong_source = validate_shard_completion(
        report,
        shard,
        executions,
        source_sha="source-b",
        config_sha256="config-a",
        phase_id="chart_initial",
        shard_id=1,
        repeats_per_direction=3,
    )
    assert wrong_source["checks"]["source_sha"] is False


def test_merge_fails_closed_on_missing_or_duplicate_shards() -> None:
    registry = build_audit_shard_registry(
        _schedules(), shard_count=3, repeats_per_direction=3
    )
    frames = {
        shard_id: _executions(registry[registry["shard_id"].eq(shard_id)])
        for shard_id in range(3)
    }
    merged = merge_validated_audit_shards(
        registry, frames, repeats_per_direction=3
    )
    assert len(merged) == len(registry) * 6
    assert not merged.duplicated(
        ["schedule_id", "direction", "repeat_index"]
    ).any()

    with pytest.raises(ValueError, match="missing shard outputs"):
        merge_validated_audit_shards(
            registry, {0: frames[0], 1: frames[1]}, repeats_per_direction=3
        )

    duplicated = frames.copy()
    duplicated[2] = pd.concat([frames[2], frames[2].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exact execution coverage"):
        merge_validated_audit_shards(
            registry, duplicated, repeats_per_direction=3
        )
