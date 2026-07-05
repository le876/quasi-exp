from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "filter_branch_consistent_dataset.py"
    spec = importlib.util.spec_from_file_location("filter_branch_consistent_dataset", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_distal_policy_prefers_lower_distal_preference_then_quality_tiebreak() -> None:
    mod = _load_module()
    branches = pd.DataFrame(
        [
            {"branch_id": 0, "distal_preference_score": -0.8, "max_tension": 800.0, "rms_rnorm": 0.03, "branch_size": 2},
            {"branch_id": 1, "distal_preference_score": 0.2, "max_tension": 200.0, "rms_rnorm": 0.01, "branch_size": 8},
        ]
    )

    scored = mod.score_branch_candidates(branches)
    assert int(scored.sort_values("branch_score").iloc[0]["branch_id"]) == 0

    tie = pd.DataFrame(
        [
            {"branch_id": 0, "distal_preference_score": -0.1, "max_tension": 1200.0, "rms_rnorm": 0.04, "branch_size": 2},
            {"branch_id": 1, "distal_preference_score": -0.1, "max_tension": 300.0, "rms_rnorm": 0.01, "branch_size": 2},
        ]
    )
    scored_tie = mod.score_branch_candidates(tie)
    assert int(scored_tie.sort_values("branch_score").iloc[0]["branch_id"]) == 1

