from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "run_anchor_canonical_sweep.py"
    spec = importlib.util.spec_from_file_location("run_anchor_canonical_sweep", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_variants_filters_failed_hard_gates() -> None:
    module = _load_module()
    rows = [
        {
            "variant": "bad",
            "hard_passed": False,
            "continuity_tension_p95_10mm": 100.0,
            "rms_rnorm_q95": 0.01,
            "max_tension": 900.0,
        },
        {
            "variant": "ok_b",
            "hard_passed": True,
            "continuity_tension_p95_10mm": 180.0,
            "rms_rnorm_q95": 0.02,
            "max_tension": 800.0,
        },
        {
            "variant": "ok_a",
            "hard_passed": True,
            "continuity_tension_p95_10mm": 180.0,
            "rms_rnorm_q95": 0.01,
            "max_tension": 1200.0,
        },
    ]

    ranked = module.rank_variants(rows)

    assert [r["variant"] for r in ranked] == ["ok_a", "ok_b"]


def test_second_stage_weight_uses_at_least_20() -> None:
    module = _load_module()

    assert module.second_stage_weight({"w_anchor": 10.0}) == 20.0
    assert module.second_stage_weight({"w_anchor": 20.0}) == 20.0
    assert module.second_stage_weight({"w_anchor": 30.0}) == 30.0


def test_model_acceptance_requires_no_worse_and_two_improved_splits(monkeypatch) -> None:
    module = _load_module()

    def fake_best_metrics(variant: str):
        if variant == "anchor_v1":
            return {split: {"tension_mae_n": 100.0} for split in module.SPLITS}
        return {
            "iid": {"tension_mae_n": 90.0},
            "radius": {"tension_mae_n": 94.0},
            "beta_block": {"tension_mae_n": 100.0},
            "angular_sector": {"tension_mae_n": 104.0},
        }

    monkeypatch.setattr(module, "baseline_best_metrics", fake_best_metrics)

    result = module.model_acceptance("candidate")

    assert result["passed"] is True
    assert result["no_split_worse_gt_5pct"] is True
    assert result["at_least_two_splits_improve_ge_5pct"] is True


def test_hard_gate_summary_checks_rows_components_and_physics(tmp_path: Path) -> None:
    module = _load_module()
    dataset_path = tmp_path / "dataset.parquet"
    meta_path = tmp_path / "dataset_meta.parquet"
    analysis_path = tmp_path / "analysis.json"

    rows = 8
    dataset = pd.DataFrame({"sample_id": range(rows)})
    for i in range(1, 13):
        dataset[f"tension_{i}_n"] = 500.0 + i
    dataset.to_parquet(dataset_path)

    meta = pd.DataFrame(
        {
            "sample_id": range(rows),
            "source_component": ["sobol_full"] * 2
            + ["lhs_full"] * 2
            + ["workspace_balanced"] * 2
            + ["distal_biased"] * 2,
        }
    )
    for i in range(1, 5):
        meta[f"beta{i}_rad"] = 0.01
    for i in range(5, 7):
        meta[f"beta{i}_rad"] = 0.02
    meta.to_parquet(meta_path)

    analysis_path.write_text(
        json.dumps(
            {
                "rms_rnorm": {"q95": 0.04},
                "tension_max": {"max": 900.0},
                "sat_ratio_max_t": 0.0,
            }
        ),
        encoding="utf-8",
    )

    summary = module.hard_gate_summary(
        dataset_path=dataset_path,
        meta_path=meta_path,
        analysis_path=analysis_path,
        expected_rows=rows,
        expected_counts={
            "sobol_full": 2,
            "lhs_full": 2,
            "workspace_balanced": 2,
            "distal_biased": 2,
        },
    )

    assert summary["hard_passed"] is True
    assert summary["component_counts"] == {
        "sobol_full": 2,
        "lhs_full": 2,
        "workspace_balanced": 2,
        "distal_biased": 2,
    }


def test_hard_gate_summary_fails_when_segmented_success_is_false(tmp_path: Path) -> None:
    module = _load_module()
    dataset_path = tmp_path / "dataset.parquet"
    meta_path = tmp_path / "dataset_meta.parquet"
    analysis_path = tmp_path / "analysis.json"

    rows = 4
    dataset = pd.DataFrame({"sample_id": range(rows)})
    for i in range(1, 13):
        dataset[f"tension_{i}_n"] = 500.0
    dataset.to_parquet(dataset_path)

    meta = pd.DataFrame(
        {
            "sample_id": range(rows),
            "source_component": ["sobol_full", "lhs_full", "workspace_balanced", "distal_biased"],
            "segmented_success": [True, True, False, True],
        }
    )
    for i in range(1, 5):
        meta[f"beta{i}_rad"] = 0.01
    for i in range(5, 7):
        meta[f"beta{i}_rad"] = 0.02
    meta.to_parquet(meta_path)

    analysis_path.write_text(
        json.dumps(
            {
                "rms_rnorm": {"q95": 0.04},
                "tension_max": {"max": 900.0},
                "sat_ratio_max_t": 0.0,
            }
        ),
        encoding="utf-8",
    )

    summary = module.hard_gate_summary(
        dataset_path=dataset_path,
        meta_path=meta_path,
        analysis_path=analysis_path,
        expected_rows=rows,
        expected_counts={
            "sobol_full": 1,
            "lhs_full": 1,
            "workspace_balanced": 1,
            "distal_biased": 1,
        },
    )

    assert summary["hard_passed"] is False
    assert summary["checks"]["segmented_success_all"] is False
