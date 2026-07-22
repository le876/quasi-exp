from __future__ import annotations

import importlib.util
from pathlib import Path


def _runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/analysis/run_generalized_ellipse_branch_identity_v11_3.py"
    )
    spec = importlib.util.spec_from_file_location("branch_identity_runner_v11_3", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_branch_identity_protocol_is_independent_and_never_authorizes_tube() -> None:
    runner = _runner_module()
    source_root = Path(__file__).resolve().parents[1]
    config = runner.load_config(
        source_root / "configs/generalized_ellipse_region_v11_branch_identity.yaml",
        preset="formal",
    )

    assert config["protocol_id"] == "generalized-ellipse-region-v11.3-branch-identity"
    assert config["source_protocol_id"] == "generalized-ellipse-region-v11.2-relaxed2x"
    assert config["output_root"] == "runs/generalized_ellipse_region_v11_branch_identity"
    assert config["protocol_constraints"]["tube_authorized"] is False
    assert config["protocol_constraints"]["preserve_source_phase1_failure"] is True
    assert config["candidates"]["pilot"] == [
        "A4_A2_178_r0_reverse_c0045",
        "A4_A2_143_r1_reverse_c0045",
    ]


def test_smoke_config_deep_merges_and_variant_inventory_is_complete() -> None:
    runner = _runner_module()
    source_root = Path(__file__).resolve().parents[1]
    config = runner.load_config(
        source_root / "configs/generalized_ellipse_region_v11_branch_identity.yaml",
        preset="smoke",
    )

    assert config["phase_counts"] == {"forensics": 12, "pilot": 12, "formal": 12}
    assert config["repair"]["lambda_reference"] == 2.0
    assert config["repair"]["candidate_budget"] == 2
    assert runner.variant_specs([0, 45, 90, 135]) == [
        ("repeat", "forward", 0),
        ("forward_cut0045", "forward", 45),
        ("forward_cut0090", "forward", 90),
        ("forward_cut0135", "forward", 135),
        ("reverse_cut0000", "reverse", 0),
        ("reverse_cut0045", "reverse", 45),
        ("reverse_cut0090", "reverse", 90),
        ("reverse_cut0135", "reverse", 135),
    ]
