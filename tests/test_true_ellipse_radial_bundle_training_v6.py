from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    analysis_dir = REPO_ROOT / "scripts" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    path = analysis_dir / "run_true_ellipse_radial_bundle_training_v6.py"
    spec = importlib.util.spec_from_file_location("run_true_ellipse_radial_bundle_training_v6", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dataset() -> pd.DataFrame:
    rows = []
    for radius_mm in (75.0, 80.0, 92.5, 100.0):
        for angle_idx in range(3):
            for is_centerline in (False, True):
                rows.append(
                    {
                        "sample_id": f"{radius_mm}:{angle_idx}:{is_centerline}",
                        "family_id": "fixed-family",
                        "trajectory_id": f"fixed-family@{radius_mm:g}",
                        "radius_mm": radius_mm,
                        "angle_idx": angle_idx,
                        "angle_rad": float(angle_idx),
                        "tube_offset_id": "center" if is_centerline else "edge",
                        "is_centerline": is_centerline,
                        "split": "validation" if radius_mm == 92.5 else "test" if radius_mm == 100.0 else "train",
                    }
                )
    return pd.DataFrame(rows)


def test_assignment_holds_out_whole_92p5_and_100_radii_and_excludes_training_centerlines() -> None:
    mod = _load_module()
    assignment, report = mod.make_training_assignment(_dataset())

    training = assignment[assignment["used_for_training"]]
    assert report["split_gate_pass"] is True
    assert set(training["radius_mm"]) == {75.0, 80.0}
    assert not training["is_centerline"].any()
    assert not assignment.loc[assignment["radius_mm"].isin([92.5, 100.0]), "used_for_training"].any()
    assert set(assignment.loc[assignment["split"].eq("validation"), "radius_mm"]) == {92.5}
    assert set(assignment.loc[assignment["split"].eq("test"), "radius_mm"]) == {100.0}


def test_formal_training_protocol_keeps_24_configs_five_seeds_and_xyz_only_inputs() -> None:
    mod = _load_module()
    args = mod.parse_args([])
    protocol = mod.formal_training_protocol_report(args)

    assert protocol["formal_training_protocol_gate_pass"] is True
    assert len(mod.model_configs()) == 24
    assert len(set(config.config_id for config in mod.model_configs())) == 24
    assert mod.MODEL_INPUT_COLUMNS == ["x_target_m", "y_target_m", "z_target_m"]
    assert set(mod.parse_int_csv(args.seeds)) == {20260711, 20260712, 20260713, 20260714, 20260715}
    assert {"radius_mm", "angle_idx", "family_id"}.isdisjoint(mod.MODEL_INPUT_COLUMNS)


def test_formal_seed_gate_requires_four_of_exactly_five_test_runs() -> None:
    mod = _load_module()
    four_pass = pd.DataFrame({"seed": [1, 2, 3, 4, 5], "model_gate_pass": [True, True, True, True, False]})
    three_pass = four_pass.copy()
    three_pass.loc[3, "model_gate_pass"] = False

    assert mod.aggregate_formal_seed_gate(four_pass)["stable_gate_pass"] is True
    assert mod.aggregate_formal_seed_gate(three_pass)["stable_gate_pass"] is False
    assert mod.aggregate_formal_seed_gate(four_pass.iloc[:4])["stable_gate_pass"] is False


def test_formal_model_gate_requires_both_validation_and_heldout_100mm_test() -> None:
    mod = _load_module()

    assert mod.formal_model_gate_pass(
        formal_claims_allowed=True,
        validation_gate={"stable_gate_pass": True},
        test_gate={"stable_gate_pass": True},
    ) is True
    assert mod.formal_model_gate_pass(
        formal_claims_allowed=True,
        validation_gate={"stable_gate_pass": False},
        test_gate={"stable_gate_pass": True},
    ) is False
    assert mod.formal_model_gate_pass(
        formal_claims_allowed=True,
        validation_gate={"stable_gate_pass": True},
        test_gate={"stable_gate_pass": False},
    ) is False


def test_only_formal_preset_may_evaluate_the_registered_100mm_test() -> None:
    mod = _load_module()
    formal = mod.final_evaluation_specs(mod.parse_args([]))
    smoke = mod.final_evaluation_specs(mod.parse_args(["--preset", "smoke"]))
    pilot = mod.final_evaluation_specs(mod.parse_args(["--preset", "pilot"]))

    assert [spec["label"] for spec in formal] == ["validation_92p5", "test_100"]
    assert [spec["label"] for spec in smoke] == ["validation_92p5"]
    assert [spec["label"] for spec in pilot] == ["validation_92p5"]


def test_registered_holdout_radii_cannot_be_overridden_from_the_cli() -> None:
    mod = _load_module()

    with pytest.raises(SystemExit):
        mod.parse_args(["--validation-radius-mm", "90"])
    with pytest.raises(SystemExit):
        mod.parse_args(["--test-radius-mm", "99"])


def test_training_authorization_requires_formal_upstream_dataset_gate() -> None:
    mod = _load_module()

    assert mod.upstream_training_authorized(
        {"formal_dataset_gate_pass": True, "training_only_support_100mm_pass": True}
    ) is True
    assert mod.upstream_training_authorized(
        {"formal_dataset_gate_pass": False, "training_only_support_100mm_pass": True}
    ) is False
    assert mod.upstream_training_authorized(
        {"formal_dataset_gate_pass": True, "training_only_support_100mm_pass": False}
    ) is False


def test_downstream_phase_reuses_current_compatible_audit_even_without_skip_existing(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    report_dir = tmp_path / "00_audit"
    report_dir.mkdir(parents=True)
    expected = "current-audit-fingerprint"
    (report_dir / "audit_report.json").write_text(
        json.dumps(
            {
                "strategy_version": mod.TRAINING_STRATEGY_VERSION,
                "task_fingerprint": expected,
                "audit_gate_pass": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_audit_task_fingerprint", lambda _args: expected)
    monkeypatch.setattr(
        mod,
        "phase_audit",
        lambda _args: (_ for _ in ()).throw(AssertionError("audit was recomputed")),
    )

    cached = mod.ensure_audit_report(SimpleNamespace(out_dir=tmp_path, skip_existing=False))

    assert cached["task_fingerprint"] == expected
