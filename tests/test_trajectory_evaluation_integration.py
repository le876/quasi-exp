from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.run_bacra_v14_3r_retry19_fullspace_direct_student as retry19


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bacra_v14_3r_retry19_fullspace_direct_student.yaml"


def _legal_candidate(target_id: str, candidate_id: str, beta: np.ndarray) -> dict[str, object]:
    return {
        "target_id": target_id,
        "candidate_id": candidate_id,
        **{
            column: float(value)
            for column, value in zip(retry19.BETA_COLUMNS, beta, strict=True)
        },
        "solver_success": True,
        "bounds_pass": True,
        "fk_residual_mm": 0.0,
        "min_margin_deg": 1.0,
    }


def _write_trajectory_waypoints(root: Path, rows: list[dict[str, object]]) -> None:
    stage = root / retry19.STAGE_DIRS["postlock_trajectories"]
    stage.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(stage / "postlock_15_trajectory_waypoints.parquet", index=False)


def test_trajectory_teacher_uses_real_cycle_dp_after_waypoint_registry_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate order cannot replace the frozen waypoint order at the DP boundary."""

    config = retry19.load_config(CONFIG)
    trajectory_id = "shuffled_loop"
    target0, target1 = f"{trajectory_id}:00000", f"{trajectory_id}:00001"
    _write_trajectory_waypoints(
        tmp_path,
        [
            {
                "trajectory_id": trajectory_id,
                "waypoint_index": 1,
                "shape_class": "ellipse",
                "trajectory_role": "heldout",
                "x_m": 1.20,
                "y_m": 0.002,
                "z_m": 0.001,
            },
            {
                "trajectory_id": trajectory_id,
                "waypoint_index": 0,
                "shape_class": "ellipse",
                "trajectory_role": "heldout",
                "x_m": 1.21,
                "y_m": 0.001,
                "z_m": 0.002,
            },
        ],
    )

    # Deliberately reverse both waypoint and candidate order.  The zero beta
    # candidate is the unique unary minimum for each target, so the expected
    # DP answer and its waypoint order are deterministic.
    candidates = pd.DataFrame(
        [
            _legal_candidate(target1, "one_far", np.full(6, 0.04)),
            _legal_candidate(target0, "zero_far", np.full(6, 0.03)),
            _legal_candidate(target1, "one_zero", np.zeros(6)),
            _legal_candidate(target0, "zero_zero", np.zeros(6)),
        ]
    )
    monkeypatch.setattr(retry19, "_gate", lambda *_args: {"status": "complete"})
    monkeypatch.setattr(retry19, "_verify_lock", lambda *_args: True)
    monkeypatch.setattr(
        retry19.retry17,
        "_solve_candidates",
        lambda *_args, **_kwargs: candidates.copy(),
    )

    gate = retry19.stage_trajectory_teacher(config, tmp_path, smoke=False)

    assert gate["status"] == "complete"
    labels = pd.read_parquet(
        tmp_path
        / retry19.STAGE_DIRS["trajectory_teacher"]
        / "trajectory_teacher_labels.parquet"
    )
    assert labels["target_id"].tolist() == [target0, target1]
    assert labels["waypoint_index"].tolist() == [0, 1]
    assert labels["candidate_id"].tolist() == ["zero_zero", "one_zero"]
    assert labels["teacher"].eq("retry17_cycle_dp").all()


def test_trajectory_teacher_rejects_nonunique_waypoint_target_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The candidate-to-waypoint association must remain many-to-one."""

    config = retry19.load_config(CONFIG)
    _write_trajectory_waypoints(
        tmp_path,
        [
            {
                "trajectory_id": "duplicate",
                "waypoint_index": 0,
                "shape_class": "ellipse",
                "trajectory_role": "heldout",
                "x_m": 1.20,
                "y_m": 0.0,
                "z_m": 0.0,
            },
            {
                "trajectory_id": "duplicate",
                "waypoint_index": 0,
                "shape_class": "ellipse",
                "trajectory_role": "heldout",
                "x_m": 1.21,
                "y_m": 0.0,
                "z_m": 0.0,
            },
        ],
    )
    duplicate_target = "duplicate:00000"
    monkeypatch.setattr(retry19, "_gate", lambda *_args: {"status": "complete"})
    monkeypatch.setattr(retry19, "_verify_lock", lambda *_args: True)
    monkeypatch.setattr(
        retry19.retry17,
        "_solve_candidates",
        lambda *_args, **_kwargs: pd.DataFrame(
            [_legal_candidate(duplicate_target, "candidate", np.zeros(6))]
        ),
    )

    with pytest.raises(pd.errors.MergeError, match="many-to-one"):
        retry19.stage_trajectory_teacher(config, tmp_path, smoke=False)


class _ReachabilityModel:
    def __init__(self, reachable_xyz: np.ndarray, reachable_beta: np.ndarray) -> None:
        self._xyz = np.asarray(reachable_xyz, dtype=float)
        self._beta = np.asarray(reachable_beta, dtype=float)

    def __call__(self, xyz: np.ndarray, *, training: bool) -> np.ndarray:
        assert training is False
        points = np.asarray(xyz, dtype=float)
        result = np.zeros((len(points), 6), dtype=float)
        for index, point in enumerate(points):
            distances = np.linalg.norm(self._xyz - point, axis=1)
            match = int(np.argmin(distances))
            if distances[match] <= 2.0e-6:
                # Force the actual two-step DLS path while staying close enough
                # for the real FK correction to succeed.
                result[index] = self._beta[match] + 1.0e-4
        return result


def test_trajectory_evaluation_uses_real_dls_fk_and_path_metrics_for_full_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = retry19.load_config(CONFIG)
    environment = retry19._environment(config)
    zero = np.asarray(environment.fk(np.zeros(6)), dtype=float).reshape(3)
    reachable_beta = np.asarray(
        [
            [0.010, -0.008, 0.012, -0.007, 0.009, -0.006],
            [0.012, -0.006, 0.011, -0.005, 0.007, -0.004],
            [0.009, -0.005, 0.010, -0.006, 0.008, -0.005],
        ]
    )
    reachable_xyz = np.asarray(environment.fk(reachable_beta), dtype=float).reshape(-1, 3)
    unreachable_xyz = np.asarray(
        [
            zero + [0.45, 0.35, 0.30],
            zero + [0.50, -0.30, 0.35],
            zero + [0.40, 0.30, -0.40],
        ],
        dtype=float,
    )
    _write_trajectory_waypoints(
        tmp_path,
        [
            {
                "trajectory_id": "reachable",
                "waypoint_index": index,
                "shape_class": "ellipse",
                "trajectory_role": "heldout",
                "x_m": point[0],
                "y_m": point[1],
                "z_m": point[2],
            }
            for index, point in enumerate(reachable_xyz)
        ]
        + [
            {
                "trajectory_id": "unreachable",
                "waypoint_index": index,
                "shape_class": "ellipse",
                "trajectory_role": "heldout",
                "x_m": point[0],
                "y_m": point[1],
                "z_m": point[2],
            }
            for index, point in enumerate(unreachable_xyz)
        ],
    )
    lock_stage = tmp_path / retry19.STAGE_DIRS["student_lock"]
    lock_stage.mkdir(parents=True)
    (lock_stage / "dataset_teacher_split_model_lock.json").write_text("{}\n", encoding="utf-8")
    model = _ReachabilityModel(reachable_xyz, reachable_beta)
    monkeypatch.setattr(retry19, "_gate", lambda *_args: {"status": "complete"})
    monkeypatch.setattr(retry19, "_verify_lock", lambda *_args: True)
    monkeypatch.setattr(
        retry19,
        "_load_selected_models",
        lambda *_args: {int(config["student"]["primary_seed"]): model},
    )

    gate = retry19.stage_trajectory_evaluation(config, tmp_path, smoke=False)

    assert gate["status"] == "complete"
    assert gate["dls2_axis_green"] is False
    report = pd.read_parquet(
        tmp_path
        / retry19.STAGE_DIRS["trajectory_evaluation"]
        / "trajectory_report.parquet"
    ).set_index("trajectory_id")
    assert bool(report.loc["reachable", "dls2_green"])
    assert report.loc["reachable", "dls2_success_rate"] == pytest.approx(1.0)
    assert not bool(report.loc["unreachable", "dls2_green"])
    assert report.loc["unreachable", "dls2_success_rate"] < float(
        config["trajectories"]["dls2_success_minimum"]
    )

    points = pd.read_parquet(
        tmp_path
        / retry19.STAGE_DIRS["trajectory_evaluation"]
        / "trajectory_waypoint_evaluation.parquet"
    )
    assert len(points) == len(reachable_xyz) + len(unreachable_xyz)
    failed = points[points["trajectory_id"].eq("unreachable")]
    failed_beta = failed.loc[:, [f"dls2_{name}" for name in retry19.BETA_COLUMNS]].to_numpy(float)
    failed_xyz = failed.loc[:, retry19.XYZ_COLUMNS].to_numpy(float)
    failed_residual_mm = np.linalg.norm(
        np.asarray(environment.fk(failed_beta), dtype=float).reshape(-1, 3) - failed_xyz,
        axis=1,
    ) * 1000.0
    expected_rate = float(
        np.mean(
            failed_residual_mm
            <= float(config["trajectories"]["dls2_fk_p95_maximum_mm"])
        )
    )
    assert report.loc["unreachable", "dls2_success_rate"] == pytest.approx(expected_rate)
