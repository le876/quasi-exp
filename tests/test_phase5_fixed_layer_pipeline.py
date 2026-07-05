from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "pipelines" / "run_phase5_fixed_layer_experiments.py"
    spec = importlib.util.spec_from_file_location("run_phase5_fixed_layer_experiments", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rank_fixed_layers_excludes_hard_gate_failures_from_normal_selection() -> None:
    mod = _load_module()
    summaries = [
        {
            "layer_label": "bad_gate",
            "gate": {
                "hard_gate_passed": False,
                "all10_theta_p95_deg": 0.5,
                "all10_tension_p95_n": 50.0,
                "beta_close_tension_p95_n": 50.0,
                "multi_branch_ball_ratio": 0.0,
                "xyz_nn_tension_mae_n": 10.0,
                "beta_nn_tension_mae_n": 10.0,
            },
        },
        {
            "layer_label": "good",
            "gate": {
                "hard_gate_passed": True,
                "all10_theta_p95_deg": 1.0,
                "all10_tension_p95_n": 100.0,
                "beta_close_tension_p95_n": 80.0,
                "multi_branch_ball_ratio": 0.1,
                "xyz_nn_tension_mae_n": 30.0,
                "beta_nn_tension_mae_n": 18.0,
            },
        },
    ]

    ranked = mod.rank_fixed_layers(summaries)

    assert [item["layer_label"] for item in ranked] == ["good", "bad_gate"]
    assert ranked[1]["phase5_score"] > ranked[0]["phase5_score"]


def test_pipeline_does_not_expand_20k_without_relabel_silver_gate() -> None:
    mod = _load_module()
    failed = {
        "gate": {
            "hard_gate_passed": True,
            "rms_rnorm_q95": 0.04,
            "max_tension_n": 1500.0,
            "all10_tension_p95_n": 95.0,
            "beta_close_tension_p95_n": 80.0,
            "same_beta_tension_p95_n": 55.0,
            "xyz_nn_tension_mae_n": 25.0,
            "beta_nn_tension_mae_n": 16.0,
        }
    }
    passed = {
        "gate": {
            **failed["gate"],
            "all10_tension_p95_n": 85.0,
            "beta_close_tension_p95_n": 65.0,
        }
    }

    assert not mod.relabel_passes_silver(failed)
    assert mod.relabel_passes_silver(passed)
