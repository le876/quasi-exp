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


def test_chart_b_seed_cleanup_accepts_chart_id_schema(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    import numpy as np
    import pandas as pd

    from scripts.analysis import run_bacra_v12_14_region_growth as v14
    from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS

    source = tmp_path / "chart_b.parquet"
    frame = pd.DataFrame(
        {
            "family_id": ["chart_b_cycle"] * 4,
            "group_id": ["chart_b"] * 4,
            "phase_idx": np.arange(4),
            "x_m": 1.0 + np.arange(4) * 0.001,
            "y_m": 0.2,
            "z_m": 0.1,
            "chart_id": "chart_B",
            **{
                name: np.arange(4) * 0.001
                for name in BETA_COLUMNS
            },
            **{
                f"jacobian_{row}_{column}": (
                    1.0 if row == column else 0.1
                )
                for row in range(3)
                for column in range(6)
            },
        }
    )
    frame.to_parquet(source, index=False)
    output = tmp_path / "run"
    (output / "00_protocol").mkdir(parents=True)
    (output / "00_protocol/gate.json").write_text(
        json.dumps({"gate_pass": True})
    )
    monkeypatch.setattr(
        v14,
        "_source_files",
        lambda config, project_root: {"source_d3": source},
    )
    config = {
        "v12_14": {
            "chart_scope": "chart_B",
            "canonical_source_priority": {"chart_B": 0},
            "expected_canonical_seed_rows": 4,
            "region": {
                "seed_count": 2,
                "conditioning_p95_multiplier": 1.5,
            },
            "seeds": {"seed_fps": 7},
        }
    }
    report = v14.stage_seed_cleanup(config, tmp_path, output)
    assert report["gate_pass"]
    clean = pd.read_parquet(
        output / "01_seed_cleanup/canonical_seed_dataset.parquet"
    )
    assert clean["dataset_source"].eq("chart_B").all()
