from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bacra_v14_3r_retry16_omega600_annular_tube_preflight.yaml"
PROTOCOL = ROOT / "docs/protocols/33-BACRA-V14.3R-retry16-Omega600-annular-tube执行协议.md"
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_3r_retry16_annular_tube.py"
LAUNCHER = ROOT / "scripts/pipelines/run_bacra_v14_3r_retry16_annular_tube.sh"


def _module():
    spec = importlib.util.spec_from_file_location("retry16_annular_runner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_freezes_annular_denominator_and_independent_teacher_seeds() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["experiment_id"] == "bacra_v14_3r_retry16_omega600_annular_tube_preflight"
    assert config["omega600"]["axial_length_mm"] == 600.0
    assert config["annular_domain"]["comparison_power"] == 18
    assert config["annular_domain"]["final_power"] == 19
    assert config["annular_domain"]["required_circle_count"] == 9
    assert config["annular_domain"]["required_large_circle_radius_mm"] == 100.0
    assert config["teacher_seed_banks"]["full"]["seed"] == 20260901
    assert config["teacher_seed_banks"]["y_seam"]["seed"] == 20260902
    assert config["teacher_seed_banks"]["z_seam"]["seed"] == 20260903
    assert config["claims"]["diagnostic_only"]
    assert not config["claims"]["claim_bearing_run_authorized"]


def test_stage_order_closes_feasibility_before_candidates_and_dataset() -> None:
    module = _module()
    assert module.STAGE_ORDER == (
        "inventory",
        "annular_domain",
        "objective_feasibility",
        "seed_banks",
        "candidate_preflight",
        "dataset",
        "summary",
    )


def test_runner_rejects_solid_interval_regression_or_claim_authorization(tmp_path: Path) -> None:
    module = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["omega600"]["axial_length_mm"] = 200.0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="registered length|600 mm"):
        module.load_config(path)

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["claims"]["claim_bearing_run_authorized"] = True
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic-only"):
        module.load_config(path)


def test_protocol_keeps_root_connector_out_of_annular_coverage() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "R_min(u) <= rho <= R_max(u)" in text
    assert "connector rows 也不能抵扣 annular coverage" in text
    assert "九条完整 circles" in text
    assert "proposal beta" in text
    assert "diagnostic_only" in text


def test_launcher_requires_binding_and_keeps_cpu_single_thread_contract() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "BACRA_RETRY16_ANNULAR_BINDING_SHA" in text
    assert "--binding-sha" in text
    assert "CUDA_VISIBLE_DEVICES=-1" in text


def test_full_circle_audit_checks_g4_rows_and_closing_edge() -> None:
    module = _module()
    phases = np.arange(8, dtype=float) * (2.0 * np.pi / 8.0)
    expanded = pd.DataFrame(
        {
            "circle_id": ["circle"] * 8,
            "x_m": np.ones(8),
            "y_m": 0.01 * np.cos(phases),
            "z_m": 0.01 * np.sin(phases),
            "fk_residual_mm": np.zeros(8),
            **{column: np.zeros(8) for column in BETA_COLUMNS},
        }
    )
    registered = pd.DataFrame(
        {
            "circle_id": ["circle"] * 3,
            "radius_mm": [10.0] * 3,
            "full_segment_count": [8] * 3,
        }
    )
    config = {
        "annular_domain": {
            "maximum_circle_step_mm": 10.0,
            "required_circle_count": 1,
            "required_large_circle_radius_mm": 5.0,
        },
        "candidate_solver": {"beta_weights": [4, 4, 2, 2, 1, 1], "residual_maximum_mm": 3.0},
        "preflight_gates": {
            "green": {"edge_weighted_p95_maximum_deg": 3.0, "edge_raw_gt7_rate_maximum": 0.02}
        },
    }
    audit, summary = module._audit_expanded_full_circles(expanded, registered, config)
    assert audit.loc[0, "full_cycle_complete"]
    assert audit.loc[0, "closing_workspace_step_mm"] <= 10.0
    assert summary["full_cycle_pass"]
    assert summary["full_circle_expected_row_count"] == summary["full_circle_actual_row_count"] == 8

    broken_audit, broken_summary = module._audit_expanded_full_circles(expanded.iloc[:-1], registered, config)
    assert not broken_audit.loc[0, "full_cycle_complete"]
    assert not broken_summary["full_cycle_pass"]


def test_objective_feasibility_connector_denominator_matches_registry(tmp_path: Path) -> None:
    module = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["config_path"] = str(CONFIG)
    domain = tmp_path / module.STAGE_DIRS["annular_domain"]
    domain.mkdir(parents=True)
    targets = pd.DataFrame(
        [
            {
                "target_id": "exact_zero",
                "annular_coverage_eligible": False,
                "target_role": "exact_zero",
            },
            {
                "target_id": "root_connector",
                "annular_coverage_eligible": False,
                "target_role": "root_connector",
            },
            *[
                {
                    "target_id": f"circle_target_{index}",
                    "annular_coverage_eligible": True,
                    "target_role": "heldout_circle",
                }
                for index in range(9)
            ],
        ]
    )
    targets.to_parquet(domain / "selected_target_registry.parquet", index=False)
    pd.DataFrame({"circle_id": [f"circle_{index}" for index in range(9)]}).to_parquet(
        domain / "registered_circle_registry.parquet", index=False
    )
    pd.DataFrame(
        columns=["left_target_id", "right_target_id", "edge_type", "circle_id"]
    ).to_parquet(domain / "registered_explicit_edges.parquet", index=False)
    (domain / "gate.json").write_text(
        json.dumps({"target_budget": 20}), encoding="utf-8"
    )
    (domain / "completion_manifest.json").write_text("{}", encoding="utf-8")

    module.stage_objective_feasibility(config, tmp_path, smoke=True)
    denominator = json.loads(
        (tmp_path / module.STAGE_DIRS["objective_feasibility"] / "denominator_size.json").read_text(
            encoding="utf-8"
        )
    )
    connector = next(item for item in denominator["denominators"] if item["id"] == "retry16_root_connector")
    assert connector["required_count"] == connector["registry"]["row_count"] == 2
