from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "baselines" / "run_beta_first_baselines.py"
    spec = importlib.util.spec_from_file_location("run_beta_first_baselines", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_family_gate_skips_when_accuracy_below_threshold() -> None:
    mod = _load_module()
    family_summary = {"recommended_k": 16, "families": {"k16": {"classifier": {"rf_accuracy": 0.42}}}}

    decision = mod.family_gate_decision(family_summary, accuracy_gate=0.70)

    assert decision["enabled"] is False
    assert "accuracy" in decision["reason"]


def test_family_gate_enables_recommended_family_when_accuracy_passes() -> None:
    mod = _load_module()
    family_summary = {"recommended_k": 8, "families": {"k8": {"classifier": {"rf_accuracy": 0.88}}}}

    decision = mod.family_gate_decision(family_summary, accuracy_gate=0.70)

    assert decision["enabled"] is True
    assert decision["k"] == 8

