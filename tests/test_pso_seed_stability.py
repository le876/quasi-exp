from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_stability_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "eval_pso_tension_seed_stability.py"
    spec = importlib.util.spec_from_file_location("eval_pso_tension_seed_stability", mod_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stability_gate_passes_strict_thresholds() -> None:
    module = _load_stability_module()

    summary = module.evaluate_gate(
        median_pairwise_mae_n=80.0,
        p95_pairwise_mae_n=150.0,
        feasible_rate=1.0,
        median_threshold_n=100.0,
        p95_threshold_n=200.0,
    )

    assert summary["gate_pass"] is True


def test_stability_gate_fails_when_p95_is_too_large() -> None:
    module = _load_stability_module()

    summary = module.evaluate_gate(
        median_pairwise_mae_n=80.0,
        p95_pairwise_mae_n=250.0,
        feasible_rate=1.0,
        median_threshold_n=100.0,
        p95_threshold_n=200.0,
    )

    assert summary["gate_pass"] is False
