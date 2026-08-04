from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.workspace_trajectories import build_workspace_trajectory_suite


def test_workspace_trajectory_suite_has_preregistered_36_family_inventory() -> None:
    rng = np.random.default_rng(7)
    xyz = rng.uniform([1.02, -0.03, -0.03], [1.20, 0.03, 0.03], size=(4000, 3))
    capability = pd.DataFrame(xyz, columns=["x_m", "y_m", "z_m"])
    indices = np.floor(xyz / 0.01).astype(int)
    unique = np.unique(indices, axis=0)
    classification = pd.DataFrame(
        {
            "cell_ix": unique[:, 0],
            "cell_iy": unique[:, 1],
            "cell_iz": unique[:, 2],
            "domain_class": [
                "resolved_multichart" if index < 4 else "resolved_single_under_budget"
                for index in range(len(unique))
            ],
        }
    )
    historical_rows = []
    for family in range(8):
        for phase in range(12):
            historical_rows.append(
                {
                    "family_id": f"legacy_{family}",
                    "phase_idx": phase,
                    "x_m": 1.1,
                    "y_m": 0.01 * np.sin(phase),
                    "z_m": 0.01 * np.cos(phase),
                    "major_semiaxis_m": 0.5,
                }
            )
    historical = pd.DataFrame(historical_rows)

    suite = build_workspace_trajectory_suite(
        capability, classification, historical, phase_count=16
    )

    assert len(suite.catalog) == 36
    assert suite.catalog["family_type"].value_counts().to_dict() == {
        "historical_final8": 8,
        "new_ellipse_circle": 8,
        "lissajous": 6,
        "bspline_3d_closed": 6,
        "high_x_near_zero_pose": 4,
        "cross_chart": 4,
    }
    assert len(suite.targets) == 28 * 16 + 8 * 12
    assert suite.targets.groupby("family_id")["phase_idx"].min().eq(0).all()
