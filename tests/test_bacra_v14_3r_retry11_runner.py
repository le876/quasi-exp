from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_3r_retry11.py"
CONFIG = ROOT / "configs/bacra_v14_3r_retry11_dual_track_zero_bridge.yaml"
PROTOCOL = ROOT / "docs/protocols/28-BACRA-V14.3R-retry11-dual-track-zero-bridge执行协议.md"
LAUNCHER = ROOT / "scripts/pipelines/run_bacra_v14_3r_retry11.sh"


def _module():
    spec = importlib.util.spec_from_file_location("retry11_runner_for_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_order_finishes_track_a_before_track_b() -> None:
    module = _module()
    assert module.STAGE_ORDER == (
        "inventory",
        "outer_sparse_wide",
        "outer_student",
        "outer_theta",
        "outer_tension_pilot",
        "outer_tension_materialization",
        "zero_seed",
        "zero_frontier_poc",
        "zero_outer_bridge",
        "summary",
    )
    assert set(module.STAGE_RUNNERS) == set(module.STAGE_ORDER)


def test_config_freezes_exploration_thresholds() -> None:
    module = _module()
    config = module.load_config(CONFIG)
    assert config["tension"]["yellow_success_rate_min"] == 0.80
    assert config["tension"]["green_success_rate_min"] == 0.95
    assert config["tension"]["projected_200k_is_diagnostic_only"] is True
    assert config["zero_seed"]["candidate_count"] == 8
    assert config["zero_seed"]["yellow_valid_count_min"] == 2
    assert config["zero_seed"]["green_valid_count_min"] == 4
    assert config["outer_dataset"]["required_round1_new_parent_service_coverage"] == 1.0


def test_protocol_has_nonblocking_student_and_track_b_normative_text() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "Track B stages SHALL NOT be predecessors" in text
    assert "Student RED does not imply Track A Physics RED" in text
    assert "projected_200k_is_diagnostic_only" not in text or "diagnostic" in text
    assert "straight_corridor_not_verified" in text
    assert "retry11_gold_wide_silver_v1" in text


def test_config_has_no_scientific_authorization() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert not any(config["claims"].values())
    assert config["student"]["physics_authorization_depends_on_student"] is False


def test_launcher_preserves_registered_dag_and_binding_argument() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    positions = [text.index(f"  {stage}\n") for stage in _module().STAGE_ORDER]
    assert positions == sorted(positions)
    assert "--binding-sha \"$BINDING_SHA\"" in text
    assert "BACRA_RETRY11_BINDING_SHA" in text


def test_summary_code_checks_track_a_checkpoint_immutability() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "summary modified the sealed Track A checkpoint" in text
    assert '"checkpoint_immutable_after_stage5": True' in text
    assert '"formal_authorization": False' in text
    assert '"--tension-worker-root"' in text
    assert 'parser.add_argument("--tension-worker-root")' in text


def test_pdf_failure_is_presentation_only() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "presentation_failure_does_not_change_science" in text
    assert "_render_pdf_nonblocking" in text


@pytest.mark.parametrize(
    ("provenance_quality", "continuation_quality"),
    [("ExactAnchor", "Gold"), ("Wide-Silver", "Silver")],
)
def test_retry11_provenance_quality_maps_to_continuation_quality(
    provenance_quality: str, continuation_quality: str
) -> None:
    module = _module()
    row = pd.Series(
        {
            "physical_point_id": "source",
            "label_quality": provenance_quality,
            "fk_residual_mm": 0.0,
            "condition_number": 1.0,
            **{name: 0.0 for name in module.BETA_COLUMNS},
        }
    )

    candidate = module._atlas_candidate_from_row(row, node_id=-1)

    assert candidate.quality == continuation_quality
    assert np.array_equal(candidate.beta_rad, np.zeros(6))
    assert candidate.diagnostics["retry11_provenance_label_quality"] == provenance_quality


def test_retry11_unknown_retained_quality_fails_closed() -> None:
    module = _module()
    row = {
        "physical_point_id": "source",
        "label_quality": "Reject",
        **{name: 0.0 for name in module.BETA_COLUMNS},
    }
    with pytest.raises(ValueError, match="unsupported quality"):
        module._atlas_candidate_from_row(row, node_id=-1)


def test_zero_retained_identity_has_parquet_stable_string_type(tmp_path: Path) -> None:
    module = _module()
    mixed = pd.DataFrame(
        {
            "target_id": ["exact_zero_anchor", 24986],
            "task_node_id": [-1, 24986],
            "physical_point_id": ["anchor", "seed"],
        }
    )

    normalized = module._zero_retained_artifact(mixed)
    target = tmp_path / "zero_retained.parquet"
    normalized.to_parquet(target, index=False)
    restored = pd.read_parquet(target)

    assert restored["target_id"].tolist() == ["exact_zero_anchor", "24986"]
    assert restored["target_id"].dtype == object
