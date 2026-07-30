from __future__ import annotations

from pathlib import Path


def test_v12_16_freezes_chart_b_region_before_multichart_training() -> None:
    from scripts.analysis import run_bacra_v12 as v12

    root = Path(__file__).resolve().parents[1]
    config = v12.load_protocol_config(
        root / "configs/bacra_v12_16_multichart.yaml", "formal"
    )
    assert config["claim_scope"] == "simulation_chart_b_region_readiness"
    assert config["deployment_claim_gate_pass"] is False
    assert config["v12_14"]["chart_scope"] == "chart_B"
    assert config["v12_14"]["expected_canonical_seed_rows"] == 720
    assert config["v12_14"]["dense"]["formal_accepted_count"] == 20000
    assert (
        config["v12_16"]["spatial_holdout"]["sealed_holdout_fraction"]
        == 0.15
    )
    assert config["v12_16"]["parallel"]["cpu_workers"] == 12


def test_chart_b_adapter_does_not_mutate_v12_16_claim_scope() -> None:
    from scripts.analysis import run_bacra_v12 as v12
    from scripts.analysis import run_bacra_v12_14_region_growth as v14
    from scripts.analysis.run_bacra_v12_16_multichart import (
        _v14_chart_b_config,
    )

    root = Path(__file__).resolve().parents[1]
    config = v12.load_protocol_config(
        root / "configs/bacra_v12_16_multichart.yaml", "smoke"
    )
    compatible = _v14_chart_b_config(config)
    assert compatible["claim_scope"] == v14.CLAIM_SCOPE
    assert compatible["v12_14"]["chart_scope"] == "chart_B"
    assert config["claim_scope"] == "simulation_chart_b_region_readiness"


def test_smoke_keeps_spatial_seal_policy_and_uses_four_workers() -> None:
    from scripts.analysis import run_bacra_v12 as v12

    root = Path(__file__).resolve().parents[1]
    formal = v12.load_protocol_config(
        root / "configs/bacra_v12_16_multichart.yaml", "formal"
    )
    smoke = v12.load_protocol_config(
        root / "configs/bacra_v12_16_multichart.yaml", "smoke"
    )
    assert (
        smoke["v12_16"]["spatial_holdout"]
        == formal["v12_16"]["spatial_holdout"]
    )
    assert smoke["v12_16"]["parallel"]["cpu_workers"] == 4
    assert smoke["v12_14"]["dense"]["formal_accepted_count"] == 40
