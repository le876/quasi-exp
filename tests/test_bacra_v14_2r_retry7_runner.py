from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_2r_retry7.py"


def _module():
    spec = importlib.util.spec_from_file_location("bacra_v14_2r_retry7", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry7_config_freezes_anchor_gauge_and_four_of_four_gate() -> None:
    module = _module()
    config = module.base.load_config(
        ROOT / "configs/bacra_v14_2r_stitched_atlas_retry7.yaml"
    )

    assert config["protocol_version"] == "retry7"
    assert config["protocol_revision"] == "empty_primary_retry3"
    assert config["reuse_sealed_stages"] == [
        "lineage_audit",
        "kr_ablation",
        "holonomy_diagnostics",
        "gauge_kernel_selection",
        "patch07_repair",
        "four_patch_gate",
        "reach_round8",
        "confirmation",
    ]
    assert config["reuse_pre_abstention_gauge_traces"] is False
    assert config["reuse_patch07_numerical_artifacts"] is True
    assert config["parallel"] == {
        "patch_workers": 12,
        "numerical_threads_per_worker": 1,
    }
    assert config["mechanism_gate"]["minimum_passing_patches"] == 4
    assert config["canonical_anchor"]["require_anchor_selected"] is True
    assert config["gauge"]["gain_candidates"] == [0.1, 0.3]
    assert config["gauge"]["proximal_slsqp_maximum_iterations"] == 400
    assert config["gauge"]["maximum_localized_excess_edges"] == 1
    assert config["reach_round8"]["seed_a"] != config["reach_round8"]["seed_b"]


def test_retry7_stage_order_is_gate_driven_funnel() -> None:
    module = _module()

    assert module.STAGE_ORDER == (
        "inventory",
        "lineage_audit",
        "kr_ablation",
        "holonomy_diagnostics",
        "gauge_kernel_selection",
        "patch07_repair",
        "four_patch_gate",
        "reach_round8",
        "confirmation",
        "meso_bridge",
        "summary",
    )


def test_prior_stage_reference_requires_fixed_point_and_artifact_closure(
    tmp_path,
) -> None:
    module = _module()
    prior = tmp_path / "prior"
    inventory = prior / module.STAGE_DIRS["inventory"]
    stage = prior / module.STAGE_DIRS["lineage_audit"]
    inventory.mkdir(parents=True)
    stage.mkdir(parents=True)
    fixed = {
        "source_sha": "source-a",
        "config_sha256": "config-a",
        "runtime_sha256": "runtime-a",
    }
    module.base._write_json(inventory / "source_fixed_point.json", fixed)
    module.base._write_json(stage / "gate.json", {"gate_pass": True})
    record = {
        "path": "gate.json",
        "bytes": (stage / "gate.json").stat().st_size,
        "sha256": module.base.sha256_file(stage / "gate.json"),
    }
    module.base._write_json(
        stage / "completion_manifest.json",
        {
            "schema_version": 1,
            **fixed,
            "stage_name": "lineage_audit",
            "artifacts": [record],
        },
    )

    closure = module._verify_prior_stage(prior, "lineage_audit")
    assert closure["artifact_count"] == 1
    assert closure["source_sha"] == "source-a"

    module.base._write_json(stage / "gate.json", {"gate_pass": False})
    with pytest.raises(RuntimeError, match="artifact mismatch"):
        module._verify_prior_stage(prior, "lineage_audit")


def test_root_variants_preserve_anchor_and_report_actual_budget() -> None:
    module = _module()
    roots = tuple((7, f"root_{index}") for index in range(8))

    ordered, ordered_report = module.base._apply_registered_root_variant(
        roots,
        requested_root_count=5,
        leave_one_out=False,
        reorder_nonanchors=True,
    )
    dropped, dropped_report = module.base._apply_registered_root_variant(
        roots,
        requested_root_count=5,
        leave_one_out=True,
        reorder_nonanchors=False,
    )

    assert ordered[0] == roots[0]
    assert set(ordered) == set(roots[:5])
    assert ordered_report["canonical_anchor_position_frozen"] is True
    assert dropped == roots[:4]
    assert dropped_report["selected_root_count"] == 4
    assert dropped_report["root_budget_requested_count"] == 4
    assert dropped_report["root_budget_filled"] is True


def test_kr_factorial_contrasts_do_not_compare_everything_to_K4_R8() -> None:
    module = _module()
    source = RUNNER.read_text(encoding="utf-8")

    assert '("K_at_R5", "main_K1_R5", "K4_R5")' in source
    assert '("K_at_R8", "K1_R8", "K4_R8")' in source
    assert '("R_at_K1", "main_K1_R5", "K1_R8")' in source
    assert '("R_at_K4", "K4_R5", "K4_R8")' in source


def test_gauge_kernel_funnel_registers_proximal_slsqp_last() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert source.index('"C3_anchor_potential"') < source.index(
        '"C4_proximal_slsqp"'
    )
    assert '"proximal_slsqp"' in source


def test_reach_round_prefix_uses_filtered_payload_not_raw_sobol_count() -> None:
    module = _module()
    columns = ("x_m", "y_m", "z_m")
    round6 = pd.DataFrame(
        {"x_m": [0.0, 1.0], "y_m": [2.0, 3.0], "z_m": [4.0, 5.0]}
    )
    round7 = pd.concat(
        [
            round6,
            pd.DataFrame({"x_m": [6.0], "y_m": [7.0], "z_m": [8.0]}),
        ],
        ignore_index=True,
    )

    assert module._verify_reach_round_prefix(
        round6, round7, physical_columns=columns
    )
    changed = round7.copy()
    changed.loc[0, "x_m"] = 99.0
    assert not module._verify_reach_round_prefix(
        round6, changed, physical_columns=columns
    )


def _repeat_row(
    *,
    repeat_id: int,
    prefix_index: int,
    target_node_id: int,
    success: bool,
    beta0: float,
) -> dict[str, object]:
    return {
        "physical_entity_id": "patch_07:fundamental_cycle:1,2,3",
        "start_node_id": 1,
        "direction": "forward",
        "repeat_id": repeat_id,
        "prefix_index": prefix_index,
        "target_node_id": target_node_id,
        "success": success,
        **{f"beta_{index}": beta0 if index == 0 else 0.0 for index in range(6)},
    }


def test_repeat_metrics_compare_only_complete_same_endpoint_traces() -> None:
    module = _module()
    rows = []
    for prefix in (1, 2, 3):
        rows.append(
            _repeat_row(
                repeat_id=0,
                prefix_index=prefix,
                target_node_id=(2, 3, 1)[prefix - 1],
                success=True,
                beta0=0.0,
            )
        )
    # The perturbed trace stops at a different physical endpoint.  Its large
    # beta must not be reported as repeat disagreement with the closed trace.
    rows.append(
        _repeat_row(
            repeat_id=1,
            prefix_index=1,
            target_node_id=2,
            success=False,
            beta0=np.deg2rad(20.0),
        )
    )

    metrics = module._repeat_metrics(pd.DataFrame.from_records(rows))

    assert metrics["max_deg"] == 0.0
    assert metrics["complete_trace_count"] == 1
    assert metrics["incomplete_trace_count"] == 1
    assert metrics["gate_pass"] is False


def test_abstention_candidate_never_claims_critical_cycle_pass() -> None:
    module = _module()
    candidates = [
        ("C1_predictor_proximal_g0.1", object()),
        ("C4_proximal_slsqp", object()),
    ]
    metrics = [
        {
            "kernel_id": "C1_predictor_proximal_g0.1",
            "critical_gate": False,
            "all_traces_complete": True,
            "repeat_max_deg": 0.01,
            "fk_residual_max_mm": 2.9,
            "geometry_excess_edge_count": 1,
            "geometry_max_deg": 1.05,
            "wall_time_s": 50.0,
        },
        {
            "kernel_id": "C4_proximal_slsqp",
            "critical_gate": False,
            "all_traces_complete": False,
            "repeat_max_deg": 0.0,
            "fk_residual_max_mm": 2.0,
            "geometry_excess_edge_count": 0,
            "geometry_max_deg": 0.3,
            "wall_time_s": 400.0,
        },
    ]

    selected = module._select_gauge_candidate(
        candidates,
        metrics,
        maximum_localized_excess_edges=1,
    )

    assert selected is not None
    assert selected[0] == "C1_predictor_proximal_g0.1"
    assert selected[2] == "localized_abstention"
    assert metrics[0]["critical_gate"] is False


def test_patch_runner_accepts_guarded_local_abstention_policy(
    tmp_path: Path,
) -> None:
    module = _module()
    config = module.base.load_config(
        ROOT / "configs/bacra_v14_2r_stitched_atlas_retry7.yaml"
    )
    stage = tmp_path / "04_gauge_kernel_selection"
    stage.mkdir(parents=True)
    (stage / "selected_kernel.json").write_text(
        json.dumps(
            {
                "gate_pass": True,
                "proceed_to_patch07_repair": True,
                "critical_gate": False,
                "abstention_required": True,
                "kernel_id": "C1_predictor_proximal_g0.1",
                "mode": "predictor_proximal",
                "gauge_gain": 0.1,
                "maximum_gauge_step_deg": 0.25,
                "anchor_weight": 0.0,
                "cartesian_step_mm": 5.0,
                "maximum_iterations": 100,
                "damping": 0.001,
            }
        ),
        encoding="utf-8",
    )

    policy = module.base._selected_gauge_policy(
        config, tmp_path, "gauge_main_K1_R5"
    )

    assert policy is not None
    assert policy.mode == "predictor_proximal"


def test_fresh_full_certificate_does_not_force_unnecessary_abstention() -> None:
    module = _module()
    reports = {
        name: {
            "gate_pass": True,
            "certificate_gate": True,
            "geometry_gate": True,
            "solver_gate": True,
            "repeat_gate": True,
            "abstention_ratio": 0.0,
            "canonical_anchor_component_selected": True,
        }
        for name in (
            "gauge_main_K1_R5",
            "gauge_task_graph_refined",
            "gauge_K1_R8",
        )
    }

    checks = module._patch07_repair_checks(
        {"abstention_required": True},
        reports,
        ordinary_refined={"gate_pass": True},
        root_budget={"gate_pass": True},
    )

    assert checks["registered_local_failure_resolved"] is True
    assert all(checks.values())
