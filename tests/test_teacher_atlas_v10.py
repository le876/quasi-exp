from __future__ import annotations

import numpy as np

from quasi_exp.teacher.atlas import build_local_atlas
from quasi_exp.teacher.canonical import TeacherPolicy, TeacherTrajectory, TeacherVariant


class LinearEnvironment:
    def jacobian(self, beta):
        return np.concatenate([np.eye(3), np.zeros((3, 3))], axis=1)


def test_local_atlas_saves_jacobian_neighbors_and_predicts_tube_path() -> None:
    xyz = np.column_stack([np.linspace(0, 0.03, 4), np.zeros(4), np.zeros(4)])
    beta = np.column_stack([xyz, np.zeros((4, 3))])
    trajectory = TeacherTrajectory(
        beta_rad=beta, theta_rad=np.zeros((4, 30)), achieved_xyz_m=xyz,
        target_xyz_m=xyz, chart_id=np.zeros(4), branch_id=np.zeros(4),
        metrics={}, provenance={}, success=True,
    )
    atlas = build_local_atlas(
        trajectory, LinearEnvironment(), TeacherPolicy(variant=TeacherVariant.T4), stride=2
    )
    shifted = xyz + np.asarray([0.0, 0.001, 0.0])
    predicted = atlas.predict_path(shifted)
    assert len(atlas.charts) == 2
    assert atlas.charts[0].jacobian.shape == (3, 6)
    assert atlas.charts[0].neighbor_chart_ids == (1,)
    np.testing.assert_allclose(predicted[:, :3], shifted, atol=5e-8)
