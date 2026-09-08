"""Overlapping local charts for predictor-corrector tube continuation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from .canonical import TeacherPolicy, TeacherTrajectory, beta_rms_deg, weighted_damped_pinv


class _Environment(Protocol):
    def jacobian(self, beta: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class LocalChart:
    chart_id: int
    anchor_xyz_m: np.ndarray
    anchor_beta_rad: np.ndarray
    jacobian: np.ndarray
    weighted_pseudoinverse: np.ndarray
    validity_radius_mm: float
    condition_number: float
    neighbor_chart_ids: tuple[int, ...]

    def predict(self, xyz_m: np.ndarray) -> np.ndarray:
        return self.anchor_beta_rad + self.weighted_pseudoinverse @ (
            np.asarray(xyz_m, dtype=float) - self.anchor_xyz_m
        )


@dataclass(frozen=True)
class LocalAtlas:
    charts: tuple[LocalChart, ...]
    phase_chart_ids: np.ndarray
    overlap_gap_p95_deg: float
    overlap_gate_pass: bool

    def predict_path(self, target_xyz_m: np.ndarray) -> np.ndarray:
        target = np.asarray(target_xyz_m, dtype=float).reshape(-1, 3)
        if len(target) != len(self.phase_chart_ids):
            raise ValueError("atlas target must align with its registered phase charts")
        by_id = {chart.chart_id: chart for chart in self.charts}
        return np.vstack(
            [by_id[int(chart_id)].predict(point) for chart_id, point in zip(self.phase_chart_ids, target)]
        )


def build_local_atlas(
    trajectory: TeacherTrajectory,
    environment: _Environment,
    policy: TeacherPolicy,
    *,
    stride: int = 12,
    validity_radius_mm: float = 15.0,
) -> LocalAtlas:
    beta = np.asarray(trajectory.beta_rad, dtype=float)
    xyz = np.asarray(trajectory.achieved_xyz_m, dtype=float)
    count = len(beta)
    anchors = list(range(0, count, max(1, int(stride))))
    charts = []
    for chart_id, phase_index in enumerate(anchors):
        jacobian = np.asarray(environment.jacobian(beta[phase_index]), dtype=float)
        pinv = weighted_damped_pinv(
            jacobian, damping=policy.damping, weights=np.asarray(policy.beta_weights)
        )
        singular = np.linalg.svd(jacobian, compute_uv=False)
        neighbours = tuple(
            sorted({(chart_id - 1) % len(anchors), (chart_id + 1) % len(anchors)})
        )
        charts.append(
            LocalChart(
                chart_id=chart_id,
                anchor_xyz_m=xyz[phase_index].copy(),
                anchor_beta_rad=beta[phase_index].copy(),
                jacobian=jacobian,
                weighted_pseudoinverse=pinv,
                validity_radius_mm=float(validity_radius_mm),
                condition_number=float(singular[0] / max(singular[-1], np.finfo(float).eps)),
                neighbor_chart_ids=neighbours,
            )
        )
    anchor_xyz = np.vstack([chart.anchor_xyz_m for chart in charts])
    distance = np.linalg.norm(xyz[:, None, :] - anchor_xyz[None, :, :], axis=2)
    phase_chart_ids = np.argmin(distance, axis=1).astype(np.int64)
    gaps = []
    for chart in charts:
        for neighbour_id in chart.neighbor_chart_ids:
            if neighbour_id <= chart.chart_id:
                continue
            neighbour = charts[neighbour_id]
            midpoint = 0.5 * (chart.anchor_xyz_m + neighbour.anchor_xyz_m)
            gaps.append(beta_rms_deg(chart.predict(midpoint), neighbour.predict(midpoint)))
    p95 = float(np.percentile(gaps, 95)) if gaps else 0.0
    return LocalAtlas(
        charts=tuple(charts),
        phase_chart_ids=phase_chart_ids,
        overlap_gap_p95_deg=p95,
        overlap_gate_pass=bool(p95 <= 0.5),
    )
