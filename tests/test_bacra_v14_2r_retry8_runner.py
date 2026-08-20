from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_2r_retry8.py"


def _module():
    spec = importlib.util.spec_from_file_location("bacra_v14_2r_retry8", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry8_config_separates_legality_progression_and_formal_gates() -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_2r_retry8_partial_relay.yaml")

    assert config["protocol_version"] == "retry8"
    assert config["sources"]["retry7_root"].endswith("retry7_abstention_retry4")
    assert config["parallel"] == {
        "numerical_workers": 12,
        "numerical_threads_per_worker": 1,
    }
    assert config["relay"]["target_count"] == 4
    assert config["relay"]["minimum_count"] == 2
    assert config["relay"]["preferred_separation_mm"] == 20.0
    assert config["relay"]["minimum_separation_mm"] == 10.0
    assert config["smoke"]["minimum_strict_unique_labels"] == 3000
    assert config["exploratory_target"] == {
        "coverage_min": 0.30,
        "largest_component_min": 0.20,
    }


def test_retry8_stage_order_reuses_retry7_then_reaches_smoke_and_5k_authorization() -> None:
    module = _module()

    assert module.STAGE_ORDER == (
        "inventory",
        "frozen_partial",
        "relay_registry",
        "growth_funnel",
        "sampled_screening",
        "partial_certificate",
        "seed_labels",
        "authorization",
        "summary",
    )
    assert set(module.STAGE_RUNNERS) == set(module.STAGE_ORDER)


def test_frozen_lineage_selection_prefers_largest_qualified_singleton() -> None:
    module = _module()
    qualification = pd.DataFrame.from_records(
        [
            {
                "chart_id": "chart_001__fragment_00",
                "qualified": True,
                "support_fraction": 0.25,
                "geometry_p95_deg": 0.1,
                "geometry_max_deg": 0.2,
            },
            {
                "chart_id": "chart_004__fragment_00",
                "qualified": True,
                "support_fraction": 0.37,
                "geometry_p95_deg": 0.2,
                "geometry_max_deg": 0.4,
            },
            {
                "chart_id": "chart_006__fragment_00",
                "qualified": True,
                "support_fraction": 0.34,
                "geometry_p95_deg": 0.05,
                "geometry_max_deg": 0.1,
            },
        ]
    )

    selected = module.select_frozen_lineage_chart(qualification)

    assert selected == "chart_004__fragment_00"


def test_screening_ranking_never_selects_d0_and_keeps_p0_fallback() -> None:
    module = _module()
    reports = pd.DataFrame.from_records(
        [
            {
                "method_id": "D0_raw_retry7",
                "is_selectable": False,
                "screen_pass": True,
                "coverage_ratio": 0.80,
                "largest_component_ratio": 0.80,
            },
            {
                "method_id": "R4W32",
                "is_selectable": True,
                "screen_pass": True,
                "coverage_ratio": 0.45,
                "largest_component_ratio": 0.40,
            },
            {
                "method_id": "P0_frozen_partial_singleton",
                "is_selectable": True,
                "screen_pass": True,
                "coverage_ratio": 0.37,
                "largest_component_ratio": 0.37,
            },
        ]
    )

    ranked = module.rank_screened_methods(reports, maximum_candidates=3)

    assert ranked[0] == "R4W32"
    assert "D0_raw_retry7" not in ranked
    assert "P0_frozen_partial_singleton" in ranked


def test_r8_growth_is_triggered_by_frontier_even_without_two_percent_gain() -> None:
    module = _module()

    assert module.should_run_r8_growth(
        r4w16_gain_over_p0=0.004,
        r4w32_gain_over_origin=0.006,
        expandable_frontier_ratio=0.06,
    )
    assert not module.should_run_r8_growth(
        r4w16_gain_over_p0=0.004,
        r4w32_gain_over_origin=0.006,
        expandable_frontier_ratio=0.04,
    )


def test_retry8_json_boundary_serializes_path_evidence(tmp_path: Path) -> None:
    module = _module()
    target = tmp_path / "closure.json"

    module._write_json(target, {"artifact_path": tmp_path / "artifact.parquet"})

    assert module._read_json(target)["artifact_path"] == str(
        tmp_path / "artifact.parquet"
    )
