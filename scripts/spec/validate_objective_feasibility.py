#!/usr/bin/env python3
"""Validate the reusable 00_objective_feasibility artifact contract.

This is a structural and arithmetic validator. It deliberately does not read
scientific registries or infer reachability, solver success, or experiment Gates.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import sys
from typing import Any


REQUIRED_FILES = (
    "objective_contract.json",
    "denominator_size.json",
    "reusable_evidence.json",
    "budget_lower_bound.json",
    "atomic_objective_schedule.json",
    "gate.json",
)
OBJECTIVE_KINDS = {"coverage", "complete_trajectory", "service_radius", "dataset_size"}
LOWER_BOUND_METHODS = {"count_credit", "registered_atomic_set_union", "not_bounded"}
GATE_STATUSES = {"feasible", "diagnostic_only", "invalid"}
ELIGIBLE_COUNT_KEYS = {
    "supervision_vertices",
    "connector_only_vertices",
    "served_coverage_units",
    "complete_trajectories",
    "verified_edges",
    "second_parent_certifications",
}
INELIGIBLE_COUNT_KEYS = {"proposal_only", "branch_conflicts", "unused"}


def minimum_new_units(
    *,
    target_units: int,
    reusable_eligible_units: int,
    maximum_credit_per_new_unit: int,
) -> int:
    """Return the optimistic number of new units needed for a count objective."""

    if min(target_units, reusable_eligible_units) < 0:
        raise ValueError("unit counts must be non-negative")
    if maximum_credit_per_new_unit <= 0:
        raise ValueError("maximum_credit_per_new_unit must be positive")
    deficit = max(0, target_units - reusable_eligible_units)
    return math.ceil(deficit / maximum_credit_per_new_unit)


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing required artifact: {path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {path.name}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name} must contain a JSON object")
        return {}
    return value


def _non_negative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _registry_is_bound(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and isinstance(value.get("sha256"), str)
        and len(value["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in value["sha256"].lower())
    )


def _is_hex_digest(value: Any, lengths: set[int]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def validate_contract_dir(root: Path) -> list[str]:
    errors: list[str] = []
    docs = {name: _load_json(root / name, errors) for name in REQUIRED_FILES}
    if errors:
        return errors

    experiment_ids = {doc.get("experiment_id") for doc in docs.values()}
    if len(experiment_ids) != 1 or not next(iter(experiment_ids), None):
        errors.append("all artifacts must use one non-empty experiment_id")
    for name, doc in docs.items():
        if doc.get("schema_version") != 1:
            errors.append(f"{name}: schema_version must be 1")

    objective = docs["objective_contract.json"]
    if objective.get("scope") != "coverage_trajectory":
        errors.append("objective_contract.json: scope must be coverage_trajectory")
    if not _is_hex_digest(objective.get("scientific_source_sha"), {40, 64}):
        errors.append("objective_contract.json: scientific_source_sha must be a Git/SHA digest")
    if not _is_hex_digest(objective.get("config_sha256"), {64}):
        errors.append("objective_contract.json: config_sha256 must be a SHA-256 hex string")
    primary = objective.get("primary_objectives")
    if not isinstance(primary, list) or not primary:
        errors.append("objective_contract.json: primary_objectives must be non-empty")
        primary = []
    objective_ids: set[str] = set()
    required_ids: set[str] = set()
    objectives_by_id: dict[str, Mapping[str, Any]] = {}
    trajectory_required = False
    for item in primary:
        if not isinstance(item, Mapping):
            errors.append("primary_objectives entries must be objects")
            continue
        objective_id = item.get("id")
        if not isinstance(objective_id, str) or not objective_id or objective_id in objective_ids:
            errors.append("primary objective ids must be non-empty and unique")
            continue
        objective_ids.add(objective_id)
        objectives_by_id[objective_id] = item
        if item.get("kind") not in OBJECTIVE_KINDS:
            errors.append(f"{objective_id}: unsupported objective kind")
        target = item.get("target")
        if (
            not isinstance(target, Mapping)
            or target.get("operator") not in {">=", "<="}
            or target.get("unit") not in {"fraction", "count", "mm"}
            or not _non_negative_number(target.get("value"))
        ):
            errors.append(f"{objective_id}: target requires operator, value, and unit")
        if item.get("required_for_claim") is True:
            required_ids.add(objective_id)
            trajectory_required |= item.get("kind") == "complete_trajectory"

    denominator = docs["denominator_size.json"]
    denominators = denominator.get("denominators")
    denominator_ids: set[str] = set()
    denominator_counts: dict[str, int] = {}
    if denominator.get("frozen_before_launch") is not True:
        errors.append("denominator_size.json: frozen_before_launch must be true")
    if not isinstance(denominators, list) or not denominators:
        errors.append("denominator_size.json: denominators must be non-empty")
        denominators = []
    for item in denominators:
        if not isinstance(item, Mapping):
            errors.append("denominator entries must be objects")
            continue
        denominator_id = item.get("id")
        if not isinstance(denominator_id, str) or not denominator_id:
            errors.append("denominator ids must be non-empty")
            continue
        if denominator_id in denominator_ids:
            errors.append(f"duplicate denominator id: {denominator_id}")
        denominator_ids.add(denominator_id)
        required_count = item.get("required_count")
        registry = item.get("registry")
        if not isinstance(required_count, int) or isinstance(required_count, bool) or required_count < 0:
            errors.append(f"{denominator_id}: required_count must be a non-negative integer")
        else:
            denominator_counts[denominator_id] = required_count
        if not _registry_is_bound(registry):
            errors.append(f"{denominator_id}: registry path and SHA-256 are required")
        if isinstance(registry, Mapping) and registry.get("row_count") != required_count:
            errors.append(f"{denominator_id}: registry.row_count must equal required_count")
    for item in primary:
        if isinstance(item, Mapping) and item.get("denominator_id") not in denominator_ids:
            errors.append(f"{item.get('id', '<unknown>')}: denominator_id is not frozen")
    resource_denominators = denominator.get("resource_denominators")
    required_resource_denominators = {
        "required_supervision_vertex_count",
        "required_logical_edge_count",
        "required_second_parent_certification_count",
    }
    if not isinstance(resource_denominators, Mapping) or not required_resource_denominators <= set(
        resource_denominators
    ):
        errors.append("denominator_size.json: required resource denominators are missing")
    elif any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in resource_denominators.values()
    ):
        errors.append("denominator_size.json: resource denominators must be non-negative integers")

    reusable = docs["reusable_evidence.json"]
    if reusable.get("proposal_beta_used_as_label_or_hint") is not False:
        errors.append("reusable_evidence.json: proposal beta must not be used as label or hint")
    if not _registry_is_bound(reusable.get("credit_registry")):
        errors.append("reusable_evidence.json: credit_registry path and SHA-256 are required")
    source_artifacts = reusable.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts or not all(
        _registry_is_bound(item) for item in source_artifacts
    ):
        errors.append("reusable_evidence.json: source_artifacts must be provenance-bound")
    for section, required_keys in (
        ("eligible_counts", ELIGIBLE_COUNT_KEYS),
        ("ineligible_counts", INELIGIBLE_COUNT_KEYS),
    ):
        values = reusable.get(section)
        if not isinstance(values, Mapping) or not required_keys <= set(values) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values.values()
        ):
            errors.append(
                f"reusable_evidence.json: {section} must contain all required non-negative counts"
            )

    lower = docs["budget_lower_bound.json"]
    lower_bounds = lower.get("objective_lower_bounds")
    bounded_ids: set[str] = set()
    lower_bound_ids: set[str] = set()
    per_objective_resource_minima: dict[str, list[float]] = {}
    if not isinstance(lower_bounds, list):
        errors.append("budget_lower_bound.json: objective_lower_bounds must be a list")
        lower_bounds = []
    for item in lower_bounds:
        if not isinstance(item, Mapping):
            errors.append("objective_lower_bounds entries must be objects")
            continue
        objective_id = item.get("objective_id")
        method = item.get("method")
        if objective_id not in objective_ids:
            errors.append(f"unknown lower-bound objective_id: {objective_id}")
            continue
        if objective_id in lower_bound_ids:
            errors.append(f"duplicate lower-bound objective_id: {objective_id}")
            continue
        lower_bound_ids.add(objective_id)
        if method not in LOWER_BOUND_METHODS:
            errors.append(f"{objective_id}: unsupported lower-bound method")
            continue
        if method != "not_bounded":
            bounded_ids.add(objective_id)
            if not _registry_is_bound(item.get("basis_registry")):
                errors.append(f"{objective_id}: bounded objective requires basis_registry")
        minimum_resources = item.get("minimum_resources")
        if not isinstance(minimum_resources, Mapping) or any(
            not _non_negative_number(value) for value in minimum_resources.values()
        ):
            errors.append(f"{objective_id}: minimum_resources must contain non-negative numbers")
        else:
            for resource_id, value in minimum_resources.items():
                per_objective_resource_minima.setdefault(resource_id, []).append(float(value))
        if method == "count_credit":
            objective_item = objectives_by_id[objective_id]
            required_count = denominator_counts.get(objective_item.get("denominator_id"))
            target = objective_item.get("target", {})
            target_value = target.get("value") if isinstance(target, Mapping) else None
            target_unit = target.get("unit") if isinstance(target, Mapping) else None
            try:
                if item["required_units"] != required_count:
                    raise ValueError("required_units does not match frozen denominator")
                if target_unit == "fraction":
                    if not 0 <= target_value <= 1:
                        raise ValueError("fraction target must be between zero and one")
                    expected_target_units = math.ceil(required_count * target_value)
                elif target_unit == "count":
                    expected_target_units = math.ceil(target_value)
                else:
                    raise ValueError("count_credit requires fraction or count target")
                if item["target_units"] != expected_target_units:
                    raise ValueError("target_units does not match objective target")
                computed = minimum_new_units(
                    target_units=item["target_units"],
                    reusable_eligible_units=item["reusable_eligible_units"],
                    maximum_credit_per_new_unit=item[
                        "maximum_credit_per_new_supervision_vertex"
                    ],
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{objective_id}: invalid count_credit inputs: {exc}")
            else:
                if minimum_resources.get("new_supervision_vertices") != computed:
                    errors.append(
                        f"{objective_id}: new supervision lower bound must equal {computed}"
                    )
    all_bounded = required_ids <= bounded_ids
    if lower.get("all_required_objectives_bounded") is not all_bounded:
        errors.append("budget_lower_bound.json: all_required_objectives_bounded is inconsistent")

    resources = lower.get("resources")
    resources_feasible = True
    if not isinstance(resources, Mapping) or not resources:
        errors.append("budget_lower_bound.json: resources must be non-empty")
        resources = {}
        resources_feasible = False
    for resource_id, values in resources.items():
        if not isinstance(values, Mapping):
            errors.append(f"resource {resource_id} must be an object")
            resources_feasible = False
            continue
        minimum = values.get("optimistic_minimum")
        budget = values.get("registered_budget")
        if not _registry_is_bound(values.get("basis_registry")):
            errors.append(f"resource {resource_id} requires basis_registry")
            resources_feasible = False
        if not _non_negative_number(minimum) or not _non_negative_number(budget):
            errors.append(f"resource {resource_id} requires non-negative minimum and budget")
            resources_feasible = False
        elif minimum > budget:
            resources_feasible = False
        if resource_id in per_objective_resource_minima and _non_negative_number(minimum):
            if minimum < max(per_objective_resource_minima[resource_id]):
                errors.append(
                    f"resource {resource_id} optimistic minimum understates an objective lower bound"
                )
                resources_feasible = False
    missing_resources = set(per_objective_resource_minima) - set(resources)
    if missing_resources:
        errors.append(
            "budget_lower_bound.json: missing top-level resources: "
            + ", ".join(sorted(missing_resources))
        )
        resources_feasible = False
    resources_feasible = resources_feasible and all_bounded
    if lower.get("all_required_resources_feasible") is not resources_feasible:
        errors.append("budget_lower_bound.json: all_required_resources_feasible is inconsistent")

    schedule = docs["atomic_objective_schedule.json"]
    if schedule.get("row_count_is_stop_condition") is not False:
        errors.append("atomic_objective_schedule.json: row count cannot be a stop condition")
    scheduled_ids = schedule.get("scheduled_objective_ids")
    if not isinstance(scheduled_ids, list):
        errors.append("atomic_objective_schedule.json: scheduled_objective_ids must be a list")
        scheduled_set: set[str] = set()
    else:
        scheduled_set = set(scheduled_ids)
        unknown_scheduled = scheduled_set - objective_ids
        if unknown_scheduled:
            errors.append(
                "atomic_objective_schedule.json: unknown scheduled objectives: "
                + ", ".join(sorted(unknown_scheduled))
            )
    schedule_complete = required_ids <= scheduled_set
    entries = schedule.get("entries")
    if not isinstance(entries, list):
        errors.append("atomic_objective_schedule.json: entries must be a list")
        entries = []
    entry_objective_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            errors.append("atomic schedule entries must be objects")
            continue
        entry_objective_id = entry.get("objective_id")
        if entry_objective_id not in objective_ids:
            errors.append(f"atomic schedule entry has unknown objective_id: {entry_objective_id}")
            continue
        entry_objective_ids.add(entry_objective_id)
        if not _registry_is_bound(entry.get("requirement_registry")):
            errors.append(f"atomic schedule entry {entry.get('id')} lacks requirement_registry")
    schedule_complete = schedule_complete and required_ids <= entry_objective_ids
    if trajectory_required and not any(
        isinstance(entry, Mapping)
        and entry.get("objective_id") in required_ids
        and entry.get("kind") == "trajectory"
        and entry.get("full_cycle_required") is True
        and entry.get("held_out") is True
        and _non_negative_number(entry.get("minimum_diameter_mm"))
        and entry.get("minimum_diameter_mm") > 0
        and _registry_is_bound(entry.get("requirement_registry"))
        for entry in entries
    ):
        schedule_complete = False
        errors.append("required complete trajectory lacks a held-out full-cycle atomic entry")

    gate = docs["gate.json"]
    inputs_valid = gate.get("inputs_valid") is True
    objective_budget_feasible = resources_feasible and schedule_complete
    if gate.get("required_objectives_budget_feasible") is not objective_budget_feasible:
        errors.append("gate.json: required_objectives_budget_feasible is inconsistent")
    diagnostic_allowed = objective.get("diagnostic_pilot_allowed") is True
    if not inputs_valid:
        expected_status = "invalid"
        expected_claim = False
        expected_diagnostic = False
    elif objective_budget_feasible:
        expected_status = "feasible"
        expected_claim = True
        expected_diagnostic = diagnostic_allowed
    else:
        expected_status = "diagnostic_only"
        expected_claim = False
        expected_diagnostic = diagnostic_allowed
    if gate.get("status") not in GATE_STATUSES or gate.get("status") != expected_status:
        errors.append(f"gate.json: status must be {expected_status}")
    if gate.get("claim_bearing_run_authorized") is not expected_claim:
        errors.append("gate.json: claim-bearing authorization is inconsistent")
    if gate.get("diagnostic_pilot_authorized") is not expected_diagnostic:
        errors.append("gate.json: diagnostic-pilot authorization is inconsistent")
    if expected_status != "feasible" and not gate.get("reason_codes"):
        errors.append("gate.json: non-feasible status requires reason_codes")
    failed_ids = gate.get("failed_objective_ids")
    if not isinstance(failed_ids, list) or not set(failed_ids) <= required_ids:
        errors.append("gate.json: failed_objective_ids must reference required objectives")
    if expected_status == "diagnostic_only" and not failed_ids:
        errors.append("gate.json: diagnostic_only requires failed_objective_ids")
    if expected_status == "feasible" and (failed_ids or gate.get("reason_codes")):
        errors.append("gate.json: feasible status cannot report failed objectives or reasons")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract_dir", type=Path)
    args = parser.parse_args(argv)
    errors = validate_contract_dir(args.contract_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("objective feasibility contract: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
