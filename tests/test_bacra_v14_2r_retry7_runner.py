from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


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
    assert config["parallel"] == {
        "patch_workers": 12,
        "numerical_threads_per_worker": 1,
    }
    assert config["mechanism_gate"]["minimum_passing_patches"] == 4
    assert config["canonical_anchor"]["require_anchor_selected"] is True
    assert config["gauge"]["gain_candidates"] == [0.1, 0.3]
    assert config["gauge"]["proximal_slsqp_maximum_iterations"] == 400
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
