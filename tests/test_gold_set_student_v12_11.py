from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS
from quasi_exp.teacher.gold_set_student import assign_cyclic_gold_section


def _row(
    phase: int, candidate: int, value_deg: float
) -> dict[str, object]:
    beta = np.full(6, np.deg2rad(value_deg), dtype=float)
    return {
        "family_id": "f0",
        "group_id": "core",
        "phase_idx": phase,
        "candidate_idx": candidate,
        "source": f"c{candidate}",
        "residual_mm": 0.1,
        "minimum_joint_margin_deg": 2.0,
        **dict(zip(BETA_COLUMNS, beta)),
    }


def test_assignment_uses_a_complete_hard_cyclic_gold_section() -> None:
    gold = pd.DataFrame(
        [
            _row(0, 0, 0.0),
            _row(0, 1, 10.0),
            _row(1, 0, 0.5),
            _row(1, 1, 10.5),
            _row(2, 0, 1.0),
            _row(2, 1, 11.0),
            _row(3, 0, 0.5),
            _row(3, 1, 10.5),
        ]
    )
    prediction = np.deg2rad(
        np.asarray(
            [[0.0] * 6, [10.5] * 6, [1.0] * 6, [10.5] * 6]
        )
    )

    selected, report = assign_cyclic_gold_section(
        prediction, gold, max_transition_deg=2.0
    )

    assert report["success"] is True
    assert report["selected_transition_max_deg"] <= 2.0
    assert selected["candidate_idx"].nunique() == 1
    assert selected["candidate_idx"].iloc[0] == 0
