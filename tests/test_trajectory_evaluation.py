from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS
from quasi_exp.teacher.trajectory_evaluation import (
    cycle_teacher,
    path_metrics,
    symmetry_prediction,
    two_step_dls,
)


class LinearForward:
    bounds = np.tile([-0.4, 0.4], (6, 1))

    def fk(self, beta: np.ndarray) -> np.ndarray:
        return np.asarray(beta).reshape(-1, 6)[:, :3]

    def jacobian(self, beta: np.ndarray) -> np.ndarray:
        return np.eye(3, 6)


def test_dls_preserves_zero_seams_bounds_and_batch_order() -> None:
    environment = LinearForward()
    xyz = np.array([[0., 0., 0.], [.1, 0., .2], [.1, .2, 0.], [.2, .15, .1], [1., .2, .1]])
    beta = np.full((len(xyz), 6), .03)
    before = beta.copy()

    corrected = two_step_dls(environment, beta, xyz, zero_xyz=np.zeros(3))

    np.testing.assert_array_equal(beta, before)
    np.testing.assert_array_equal(corrected[0], np.zeros(6))
    np.testing.assert_array_equal(corrected[1, [0, 2, 4]], np.zeros(3))
    np.testing.assert_array_equal(corrected[2, [1, 3, 5]], np.zeros(3))
    assert np.all(corrected >= environment.bounds[:, 0])
    assert np.all(corrected <= environment.bounds[:, 1])
    np.testing.assert_allclose(environment.fk(corrected[3:4])[0], xyz[3], rtol=0, atol=1e-9)
    assert corrected[4, 0] == environment.bounds[0, 1]


@pytest.mark.parametrize("beta,xyz", [
    (np.zeros((2, 5)), np.zeros((2, 3))),
    (np.zeros((2, 6)), np.zeros((3, 3))),
    (np.zeros((2, 6)), np.zeros((2, 2))),
    (np.zeros(6), np.zeros(3)),
])
@pytest.mark.parametrize("operation", ["dls", "metrics"])
def test_array_interfaces_reject_wrong_shapes_and_row_alignment(beta, xyz, operation) -> None:
    with pytest.raises(ValueError):
        if operation == "dls":
            two_step_dls(LinearForward(), beta, xyz, zero_xyz=np.zeros(3))
        else:
            path_metrics(xyz, beta, LinearForward())


def test_symmetry_restores_signed_quadrants_and_fixed_axes() -> None:
    inputs = []

    def model(xyz, *, training):
        assert training is False
        inputs.append(xyz.copy())
        return np.tile(np.arange(1, 7, dtype=float) / 10, (len(xyz), 1))

    xyz = np.array([[1.2, .1, .2], [1.2, -.1, .2], [1.2, .1, -.2],
                    [1.2, -.1, -.2], [1.2, 0., .2], [1.2, .1, 0.], [1.2, 0., 0.]])
    before = xyz.copy()
    result = symmetry_prediction(model, xyz, np.array([1.2, 0., 0.]))

    np.testing.assert_array_equal(xyz, before)
    assert inputs[0].dtype == np.float32
    assert np.all(inputs[0][:, 1:] >= 0)
    expected = np.array([
        [.1, .2, .3, .4, .5, .6],
        [-.1, .2, -.3, .4, -.5, .6],
        [.1, -.2, .3, -.4, .5, -.6],
        [-.1, -.2, -.3, -.4, -.5, -.6],
        [0., .2, 0., .4, 0., .6],
        [.1, 0., .3, 0., .5, 0.],
        [0., 0., 0., 0., 0., 0.],
    ])
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("output", [np.zeros((2, 5)), np.zeros((1, 6))])
def test_symmetry_rejects_misaligned_model_output(output) -> None:
    def model(xyz, *, training):
        return output

    with pytest.raises(ValueError):
        symmetry_prediction(model, np.ones((2, 3)), np.zeros(3))


def candidate_bank() -> pd.DataFrame:
    rows = []
    for waypoint in (2, 0, 1):
        for name, sign in (("b", 1), ("a", -1)):
            rows.append({
                "target_id": f"t{waypoint}", "candidate_id": f"t{waypoint}:{name}",
                "waypoint_index": waypoint, "trajectory_id": "cycle",
                "solver_success": True, "bounds_pass": True,
                "fk_residual_mm": 0., "min_margin_deg": 10.,
                **{column: sign * .03 for column in BETA_COLUMNS},
            })
    return pd.DataFrame(rows)


def select_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    return cycle_teacher(frame, pairwise_lambda=4., weights=(4, 4, 2, 2, 1, 1), tau_deg=7.)


def test_cycle_teacher_keeps_order_deterministic_tie_break_and_metadata() -> None:
    candidates = candidate_bank()
    before = candidates.copy(deep=True)
    selected = select_cycle(candidates)

    assert selected.candidate_id.tolist() == ["t0:a", "t1:a", "t2:a"]
    assert selected.waypoint_index.tolist() == [0, 1, 2]
    assert selected.trajectory_id.tolist() == ["cycle"] * 3
    assert selected.teacher.tolist() == ["retry17_cycle_dp"] * 3
    assert selected.pairwise_lambda.tolist() == [4.] * 3
    pd.testing.assert_frame_equal(selected, select_cycle(candidates.sample(frac=1., random_state=5)))
    pd.testing.assert_frame_equal(candidates, before)


@pytest.mark.parametrize("column", ["waypoint_index", "candidate_id", "target_id", "beta6_rad",
                                    "solver_success", "bounds_pass", "fk_residual_mm", "min_margin_deg"])
def test_cycle_teacher_reports_missing_input_field(column: str) -> None:
    with pytest.raises(ValueError, match=column):
        select_cycle(candidate_bank().drop(columns=column))


def test_cycle_teacher_preserves_empty_and_all_illegal_results() -> None:
    candidates = candidate_bank()
    assert select_cycle(candidates.iloc[:0]).empty
    assert select_cycle(candidates.assign(solver_success=False)).empty
    assert select_cycle(candidates.assign(bounds_pass=False)).empty
    assert select_cycle(candidates.assign(fk_residual_mm=4.)).empty


def test_path_metrics_counts_closing_edge_only_for_closed_paths() -> None:
    xyz = np.zeros((3, 3))
    beta = np.zeros((3, 6))
    beta[:, 0] = [0., .1, .2]
    opened = path_metrics(xyz, beta, LinearForward(), closed=False)
    closed = path_metrics(xyz, beta, LinearForward(), closed=True)

    assert set(opened) == {"fk_p95_mm", "fk_maximum_mm", "path_step_excess_p99_mm",
                           "path_step_excess_maximum_mm", "raw_step_gt7_rate", "raw_step_maximum_deg"}
    assert opened["fk_maximum_mm"] == closed["fk_maximum_mm"] == 200.
    assert opened["path_step_excess_maximum_mm"] == 100.
    assert closed["path_step_excess_maximum_mm"] == 200.
    assert opened["raw_step_gt7_rate"] == 0.
    assert closed["raw_step_gt7_rate"] == 1 / 3
    assert "success_rate" not in closed


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_nonfinite_predictions_preserve_failure_metrics_without_fk(nonfinite) -> None:
    class UnusedForward:
        def fk(self, beta):
            raise AssertionError("non-finite predictions must not reach FK")

    beta = np.zeros((2, 6))
    beta[1, 0] = nonfinite
    result = path_metrics(np.zeros((2, 3)), beta, UnusedForward())
    assert result["raw_step_gt7_rate"] == 1.
    assert all(np.isinf(value) for key, value in result.items() if key != "raw_step_gt7_rate")
