from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.student import StudentKind, build_student_arrays, split_by_trajectory


def _dataset() -> pd.DataFrame:
    rows = []
    for trajectory in ("a", "b", "c", "d"):
        for phase in range(3):
            row = {
                "trajectory_id": trajectory,
                "target_x_m": float(phase),
                "target_y_m": 0.0,
                "target_z_m": 0.0,
                "chart_id": phase % 2,
            }
            for index in range(1, 7):
                row[f"teacher_beta{index}_rad"] = 0.1 * phase
                row[f"previous_beta{index}_rad"] = 0.1 * ((phase - 1) % 3)
            rows.append(row)
    return pd.DataFrame(rows)


def test_split_by_trajectory_has_no_trajectory_leakage() -> None:
    split = split_by_trajectory(_dataset(), validation_fraction=0.25, test_fraction=0.25, seed=7)
    roles = {
        role: set(frame["trajectory_id"].unique())
        for role, frame in split.items()
    }
    assert roles["train"].isdisjoint(roles["validation"])
    assert roles["train"].isdisjoint(roles["test"])
    assert roles["validation"].isdisjoint(roles["test"])


def test_s0_s2_s4_arrays_match_the_registered_representations() -> None:
    frame = _dataset()
    s0 = build_student_arrays(frame, StudentKind.S0)
    s2 = build_student_arrays(frame, StudentKind.S2)
    s4 = build_student_arrays(frame, StudentKind.S4)

    assert s0.features.shape == (12, 3)
    assert s0.targets.shape == (12, 6)
    assert s2.features.shape == (12, 12)  # xyz, delta xyz, previous beta
    assert s2.targets.shape == (12, 6)  # delta beta
    assert s4.features.shape == (12, 3)
    np.testing.assert_array_equal(s4.chart_labels, frame["chart_id"].to_numpy())
