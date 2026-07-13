from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "run_true_ellipse_reachability_atlas_v2.py"
    spec = importlib.util.spec_from_file_location("run_true_ellipse_reachability_atlas_v2", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_preset_smoke_values_are_bounded_for_fast_ci() -> None:
    mod = _load_runner()

    args = mod.parse_args(["--preset", "smoke", "--out-dir", "/tmp/atlas_smoke"])
    mod.apply_preset_defaults(args)

    assert args.pool_rows <= 4096
    assert args.max_centers <= 16
    assert args.phase_step_deg >= 90.0
    assert args.max_pointwise_candidates <= 3


def test_phase0_baseline_audit_writes_negative_control_summary(tmp_path: Path) -> None:
    mod = _load_runner()
    old = tmp_path / "old"
    (old / "02_support_recompute").mkdir(parents=True)
    (old / "03_centerline_active_inverse").mkdir(parents=True)
    pd.DataFrame(
        [
            {"pool_label": "a1", "ellipse_id": "E75", "true_nn_p95_mm": 86.04},
            {"pool_label": "hierarchical", "ellipse_id": "E75", "true_nn_p95_mm": 30.26},
        ]
    ).to_csv(old / "02_support_recompute" / "support_summary_by_pool.csv", index=False)
    pd.DataFrame([{"residual_p95_mm": 61.64}]).to_csv(
        old / "03_centerline_active_inverse" / "centerline_inverse_report.csv",
        index=False,
    )

    args = mod.parse_args(["--preset", "smoke", "--out-dir", str(tmp_path / "run"), "--baseline-dir", str(old)])
    mod.apply_preset_defaults(args)
    mod.phase0_baseline_audit(args)

    out = tmp_path / "run" / "00_baseline_audit" / "current_failure_summary.md"
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "negative control" in text
    assert "61.64" in text
