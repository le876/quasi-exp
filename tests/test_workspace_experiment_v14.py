from __future__ import annotations

import numpy as np

from quasi_exp.teacher.workspace_experiment import (
    ReachSamplingRoundSpec,
    cell_centers,
    frontier_candidates,
    generate_independent_reach_samples,
    tip_focused_beta,
)
from quasi_exp.teacher.workspace_reach import CellKey, WorkspaceGridSpec


class _AffineEnvironment:
    bounds = np.tile(np.asarray([[-1.0, 1.0]]), (6, 1))

    def fk(self, beta: np.ndarray) -> np.ndarray:
        values = np.asarray(beta, dtype=float).reshape(-1, 6)
        # Place every sample in the registered slab while preserving a useful
        # three-dimensional occupancy pattern.
        return np.column_stack(
            [1.1 + 0.08 * values[:, 0], 0.08 * values[:, 1], 0.08 * values[:, 2]]
        )


def test_independent_reach_samples_are_cumulative_and_seed_disjoint() -> None:
    generated = generate_independent_reach_samples(
        _AffineEnvironment(),
        _AffineEnvironment.bounds,
        grid=WorkspaceGridSpec(x_slab_m=(1.0, 1.2)),
        rounds=(
            ReachSamplingRoundSpec(power=3, seed_a=11, seed_b=12),
            ReachSamplingRoundSpec(power=4, seed_a=13, seed_b=14),
        ),
        chunk_rows=32,
    )

    assert [len(item.replica_a.xyz_m) for item in generated.rounds] == [8, 24]
    assert [len(item.replica_b.xyz_m) for item in generated.rounds] == [8, 24]
    assert generated.rounds[-1].replica_a.scramble_seeds == (11, 13)
    assert generated.rounds[-1].replica_b.scramble_seeds == (12, 14)
    assert not (
        set(generated.rounds[-1].replica_a.scramble_seeds)
        & set(generated.rounds[-1].replica_b.scramble_seeds)
    )
    assert generated.beta_a.shape == (24, 6)
    assert generated.beta_b.shape == (24, 6)


def test_tip_pool_stays_inside_registered_fraction_of_full_span() -> None:
    bounds = np.asarray(
        [[-2.0, 2.0], [-1.0, 3.0], [-4.0, 0.0], [-1.0, 1.0], [-3.0, 5.0], [0.0, 2.0]]
    )
    beta = tip_focused_beta(bounds, power=5, seed=41, scale_max=0.25)
    midpoint = bounds.mean(axis=1)
    halfspan = 0.5 * (bounds[:, 1] - bounds[:, 0])

    assert beta.shape == (32, 6)
    assert np.max(np.abs((beta - midpoint) / halfspan)) <= 0.25 + 1.0e-12


def test_frontier_candidates_are_unsupported_face_neighbours_with_stable_centres() -> None:
    grid = WorkspaceGridSpec(levels_mm=(20, 10, 5), x_slab_m=(1.0, 1.2))
    supported = frozenset(
        {
            CellKey(20, 50, 0, 0),
            CellKey(20, 51, 0, 0),
        }
    )
    candidates = frontier_candidates(
        supported,
        grid=grid,
        maximum_count=4,
    )

    assert candidates
    assert not (set(candidates) & set(supported))
    assert all(
        any(neighbour in supported for neighbour in cell.face_neighbors())
        for cell in candidates
    )
    centres = cell_centers(candidates, grid=grid)
    assert centres.shape == (len(candidates), 3)
    assert np.all((centres[:, 0] >= 1.0) & (centres[:, 0] <= 1.2))
