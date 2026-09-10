from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    ROOT
    / "scripts/spec"
    / "validate_objective_feasibility.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("objective_feasibility", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _registry(path: str, rows: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": path, "sha256": "a" * 64}
    if rows is not None:
        value.update({"id_column": "id", "row_count": rows})
    return value


def _write_fixture(
    root: Path,
    *,
    required_units: int = 100,
    target_units: int = 80,
    reusable_units: int = 20,
    budget: int = 60,
    status: str = "feasible",
    claim_authorized: bool = True,
    row_count_is_stop_condition: bool = False,
    trajectory: bool = False,
) -> None:
    root.mkdir()
    experiment_id = "fixture"
    objective_id = "large_circle" if trajectory else "shell_coverage"
    objective_kind = "complete_trajectory" if trajectory else "coverage"
    minimum = max(0, target_units - reusable_units)
    target = (
        {"operator": ">=", "value": target_units, "unit": "count"}
        if trajectory
        else {"operator": ">=", "value": target_units / required_units, "unit": "fraction"}
    )
    docs = {
        "objective_contract.json": {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "scope": "coverage_trajectory",
            "scientific_source_sha": "b" * 40,
            "config_sha256": "c" * 64,
            "diagnostic_pilot_allowed": True,
            "primary_objectives": [
                {
                    "id": objective_id,
                    "kind": objective_kind,
                    "metric": "complete_ring_count" if trajectory else "shell_cell_coverage",
                    "required_for_claim": True,
                    "denominator_id": "required_units",
                    "target": target,
                }
            ],
        },
        "denominator_size.json": {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "frozen_before_launch": True,
            "denominators": [
                {
                    "id": "required_units",
                    "kind": objective_kind,
                    "unit": "ring" if trajectory else "strict_supervision_vertex",
                    "required_count": required_units,
                    "registry": _registry("required.parquet", required_units),
                }
            ],
            "resource_denominators": {
                "required_supervision_vertex_count": required_units,
                "required_logical_edge_count": 0,
                "required_second_parent_certification_count": 0,
            },
        },
        "reusable_evidence.json": {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "source_artifacts": [_registry("upstream.parquet")],
            "eligible_counts": {
                "supervision_vertices": reusable_units,
                "connector_only_vertices": 0,
                "served_coverage_units": reusable_units,
                "complete_trajectories": 0,
                "verified_edges": 0,
                "second_parent_certifications": 0,
            },
            "ineligible_counts": {"proposal_only": 0, "branch_conflicts": 0, "unused": 0},
            "credit_registry": _registry("credit.parquet"),
            "proposal_beta_used_as_label_or_hint": False,
        },
        "budget_lower_bound.json": {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "objective_lower_bounds": [
                {
                    "objective_id": objective_id,
                    "method": "count_credit",
                    "required_units": required_units,
                    "target_units": target_units,
                    "reusable_eligible_units": reusable_units,
                    "maximum_credit_per_new_supervision_vertex": 1,
                    "minimum_resources": {"new_supervision_vertices": minimum},
                    "basis_registry": _registry("lower_bound.parquet"),
                }
            ],
            "resources": {
                "new_supervision_vertices": {
                    "optimistic_minimum": minimum,
                    "registered_budget": budget,
                    "basis_registry": _registry("resource_union.parquet"),
                }
            },
            "all_required_objectives_bounded": True,
            "all_required_resources_feasible": minimum <= budget,
        },
        "atomic_objective_schedule.json": {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "row_count_is_stop_condition": row_count_is_stop_condition,
            "scheduled_objective_ids": [objective_id],
            "entries": [
                {
                    "id": "large_circle_0" if trajectory else "coverage_batch_0",
                    "objective_id": objective_id,
                    "kind": "trajectory" if trajectory else "coverage_block",
                    "priority": 0,
                    "required_for_claim": True,
                    "minimum_diameter_mm": 500.0 if trajectory else 0.0,
                    "full_cycle_required": trajectory,
                    "held_out": trajectory,
                    "requirement_registry": _registry("atomic.parquet"),
                    "reserved_resources": {"new_supervision_vertices": minimum},
                }
            ],
        },
        "gate.json": {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "inputs_valid": True,
            "required_objectives_budget_feasible": minimum <= budget,
            "status": status,
            "claim_bearing_run_authorized": claim_authorized,
            "diagnostic_pilot_authorized": True,
            "failed_objective_ids": [] if minimum <= budget else [objective_id],
            "reason_codes": [] if minimum <= budget else [
                "NEW_SUPERVISION_BUDGET_BELOW_OPTIMISTIC_MINIMUM"
            ],
        },
    }
    for name, value in docs.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")


def test_feasible_contract_is_valid(tmp_path: Path) -> None:
    module = _module()
    contract = tmp_path / "00_objective_feasibility"
    _write_fixture(contract)
    assert module.validate_contract_dir(contract) == []


def test_retry14_scale_budget_is_diagnostic_only(tmp_path: Path) -> None:
    module = _module()
    contract = tmp_path / "00_objective_feasibility"
    required_units = 81_907
    target_units = 65_526
    reusable_units = 312
    budget = 2_000
    _write_fixture(
        contract,
        required_units=required_units,
        target_units=target_units,
        reusable_units=reusable_units,
        budget=budget,
        status="diagnostic_only",
        claim_authorized=False,
    )
    assert module.minimum_new_units(
        target_units=target_units,
        reusable_eligible_units=reusable_units,
        maximum_credit_per_new_unit=1,
    ) == 65_214
    assert module.validate_contract_dir(contract) == []
    gate = json.loads((contract / "gate.json").read_text(encoding="utf-8"))
    assert gate["claim_bearing_run_authorized"] is False


def test_infeasible_retry14_scale_cannot_claim_feasible_authorization(tmp_path: Path) -> None:
    module = _module()
    contract = tmp_path / "00_objective_feasibility"
    _write_fixture(
        contract,
        required_units=81_907,
        target_units=65_526,
        reusable_units=312,
        budget=2_000,
        status="feasible",
        claim_authorized=True,
    )
    errors = module.validate_contract_dir(contract)
    assert any("status must be diagnostic_only" in error for error in errors)
    assert any("claim-bearing authorization is inconsistent" in error for error in errors)


def test_row_count_cannot_replace_coverage_progress(tmp_path: Path) -> None:
    module = _module()
    contract = tmp_path / "00_objective_feasibility"
    _write_fixture(contract, row_count_is_stop_condition=True)
    errors = module.validate_contract_dir(contract)
    assert any("row count cannot be a stop condition" in error for error in errors)


def test_complete_trajectory_requires_held_out_full_cycle_entry(tmp_path: Path) -> None:
    module = _module()
    contract = tmp_path / "00_objective_feasibility"
    _write_fixture(contract, trajectory=True)
    schedule_path = contract / "atomic_objective_schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["entries"][0]["full_cycle_required"] = False
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    errors = module.validate_contract_dir(contract)
    assert any("held-out full-cycle atomic entry" in error for error in errors)
