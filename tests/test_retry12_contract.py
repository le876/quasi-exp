from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.teacher.retry12_symmetry import (
    BETA_COLUMNS,
    assign_quotient_macroblock_splits,
    corrected_trajectory_metrics,
    orbit_class,
    reduced_seam_beta_samples,
    select_exact_orbit_budget,
)


BOUNDS = np.asarray(
    [
        [-np.radians(5), np.radians(5)],
        [-np.radians(5), np.radians(5)],
        [-np.radians(10), np.radians(10)],
        [-np.radians(10), np.radians(10)],
        [-np.radians(15), np.radians(15)],
        [-np.radians(15), np.radians(15)],
    ]
)


def test_reduced_seam_sobol_freezes_symmetry_odd_coordinates() -> None:
    y = reduced_seam_beta_samples(BOUNDS, seam="y_seam", power=5, seed=1)
    z = reduced_seam_beta_samples(BOUNDS, seam="z_seam", power=5, seed=2)

    assert np.array_equal(y[:, [0, 2, 4]], np.zeros((len(y), 3)))
    assert np.array_equal(z[:, [1, 3, 5]], np.zeros((len(z), 3)))
    assert np.any(y[:, [1, 3, 5]] != 0.0)
    assert np.any(z[:, [0, 2, 4]] != 0.0)


def test_theta_storage_is_direct_beta_expansion_without_sign() -> None:
    beta = np.asarray([0.01, -0.02, 0.03, -0.04, 0.05, -0.06])
    theta = beta_to_theta(beta)

    assert theta.shape == (30,)
    assert theta[0] == beta[0]
    assert theta[1] == beta[1]
    assert theta[10] == beta[2]
    assert theta[29] == beta[5]
    assert not np.array_equal(theta, beta_to_theta(-beta))


def test_split_is_owned_by_quotient_macroblock_not_orbit_hash() -> None:
    frame = pd.DataFrame(
        {
            "x_m": [1.0, 1.001, 1.0, 1.001],
            "y_m": [0.02, -0.02, 0.021, -0.021],
            "z_m": [0.02, 0.02, -0.02, -0.02],
            "symmetry_orbit_id": ["a", "a", "b", "b"],
        }
    )
    assigned = assign_quotient_macroblock_splits(frame, block_size_mm=40, seed=20260882)

    assert assigned["quotient_macroblock_id"].nunique() == 1
    assert assigned["split_role"].nunique() == 1


def test_orbit_selector_satisfies_19999_parity_from_start() -> None:
    rows = [
        {
            "fundamental_representative_id": "zero",
            "orbit_size": 1,
            "mandatory_seam_representative": False,
            "selection_priority": 0,
        }
    ]
    rows.extend(
        {
            "fundamental_representative_id": f"seam_{index}",
            "orbit_size": 2,
            "mandatory_seam_representative": index < 20,
            "selection_priority": index,
        }
        for index in range(100)
    )
    rows.extend(
        {
            "fundamental_representative_id": f"interior_{index}",
            "orbit_size": 4,
            "mandatory_seam_representative": False,
            "selection_priority": index,
        }
        for index in range(6000)
    )

    selected = select_exact_orbit_budget(pd.DataFrame(rows), target_expanded_rows=19999)

    assert selected["orbit_size"].sum() == 19999
    assert selected["orbit_size"].eq(2).sum() % 2 == 1
    assert set(f"seam_{index}" for index in range(20)).issubset(
        set(selected["fundamental_representative_id"])
    )


def test_nonzero_double_seam_abstains() -> None:
    zero = np.asarray([1.215498, 0.0, 0.0])

    assert orbit_class(zero, exact_tolerance_m=1.0e-9, zero_xyz_m=zero) == "exact_zero"
    assert orbit_class([1.1, 0.0, 0.0], exact_tolerance_m=1.0e-9, zero_xyz_m=zero) == "double_seam_nonzero_abstain"
    assert orbit_class([1.1, 0.0, 0.1], exact_tolerance_m=1.0e-9, zero_xyz_m=zero) == "y_seam"


def test_corrected_continuity_is_reported_separately_from_raw() -> None:
    raw = np.zeros((3, 6))
    corrected = raw.copy()
    corrected[2, 0] = np.radians(8.0)

    metrics = corrected_trajectory_metrics(raw, corrected)

    assert metrics["raw_step_gt7_rate"] == 0.0
    assert metrics["dls2_corrected_step_gt7_rate"] == 0.5
