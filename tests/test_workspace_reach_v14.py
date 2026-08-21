from __future__ import annotations

import numpy as np
import pytest

from quasi_exp.teacher.workspace_reach import (
    CellKey,
    FrontierProbeEvidence,
    ReachProxyBuilder,
    ReachProxyConfig,
    ReachReplica,
    ReachReplicaRound,
    ReachStatus,
    WorkspaceGridSpec,
)


def _replica(name: str, seeds: tuple[int, ...], xyz: list[list[float]]) -> ReachReplica:
    return ReachReplica(
        replica_id=name,
        scramble_seeds=seeds,
        xyz_m=np.asarray(xyz, dtype=float),
    )


def test_reach_proxy_requires_independent_scrambles() -> None:
    config = ReachProxyConfig(
        grid=WorkspaceGridSpec(levels_mm=(20, 10, 5), x_slab_m=(1.0, 1.2))
    )
    with pytest.raises(ValueError, match="independent scrambled"):
        same_seed = ReachReplicaRound(
            round_id=1,
            replica_a=_replica("A", (41,), [[1.01, 0.0, 0.0]]),
            replica_b=_replica("B", (41,), [[1.01, 0.0, 0.0]]),
        )
        ReachProxyBuilder(config).build((same_seed,))


def test_reach_proxy_never_certifies_an_unseen_cell_from_sampling_failure() -> None:
    config = ReachProxyConfig(
        grid=WorkspaceGridSpec(levels_mm=(20, 10, 5), x_slab_m=(1.0, 1.2))
    )
    round_1 = ReachReplicaRound(
        round_id=1,
        replica_a=_replica("A", (41,), [[1.01, 0.001, 0.001]]),
        replica_b=_replica("B", (42,), [[1.01, 0.001, 0.001]]),
    )
    unseen = CellKey(level_mm=20, ix=51, iy=0, iz=0)

    result = ReachProxyBuilder(config).build(
        (round_1,),
        frontier_evidence=(
            FrontierProbeEvidence(cell=unseen, found_valid_inverse=False),
        ),
    )

    assert result.status_by_cell[unseen] is ReachStatus.EMPIRICAL_UNSUPPORTED
    assert unseen not in result.certified_unreachable_cells
    assert result.convergence_gate is False


def test_reach_proxy_reports_jaccard_and_frontier_success_expands_lower_proxy() -> None:
    config = ReachProxyConfig(
        grid=WorkspaceGridSpec(levels_mm=(20, 10, 5), x_slab_m=(1.0, 1.2)),
        required_consecutive_rounds=1,
        minimum_weighted_jaccard=0.49,
        maximum_new_volume_ratio=1.0,
        maximum_boundary_change_ratio=1.0,
        maximum_frontier_new_volume_ratio=1.0,
    )
    # At 20 mm, A occupies x cells 50 and 51 while B occupies 50 and 52.
    round_1 = ReachReplicaRound(
        round_id=1,
        replica_a=_replica("A", (41,), [[1.001, 0.001, 0.001], [1.021, 0.001, 0.001]]),
        replica_b=_replica("B", (42,), [[1.001, 0.001, 0.001], [1.041, 0.001, 0.001]]),
    )
    frontier_cell = CellKey(level_mm=20, ix=53, iy=0, iz=0)

    result = ReachProxyBuilder(config).build(
        (round_1,),
        frontier_evidence=(
            FrontierProbeEvidence(cell=frontier_cell, found_valid_inverse=True),
        ),
    )

    metric = result.replica_metrics[-1]
    assert metric.raw_jaccard == pytest.approx(1.0 / 3.0)
    assert metric.volume_weighted_jaccard == pytest.approx(1.0 / 3.0)
    assert frontier_cell in result.proxy_lower_cells
    assert result.status_by_cell[frontier_cell] is ReachStatus.EMPIRICAL_SUPPORTED


def test_only_explicit_certificate_can_mark_unreachable() -> None:
    config = ReachProxyConfig(
        grid=WorkspaceGridSpec(levels_mm=(20, 10, 5), x_slab_m=(1.0, 1.2))
    )
    round_1 = ReachReplicaRound(
        round_id=1,
        replica_a=_replica("A", (41,), [[1.01, 0.0, 0.0]]),
        replica_b=_replica("B", (42,), [[1.01, 0.0, 0.0]]),
    )
    cell = CellKey(level_mm=20, ix=54, iy=0, iz=0)
    result = ReachProxyBuilder(config).build(
        (round_1,),
        frontier_evidence=(
            FrontierProbeEvidence(
                cell=cell,
                found_valid_inverse=False,
                unreachable_certificate="interval_fk_v1",
            ),
        ),
    )

    assert result.status_by_cell[cell] is ReachStatus.UNREACHABLE_CERTIFIED
    assert cell in result.certified_unreachable_cells


def test_registered_forward_pool_is_supplemental_support_not_frontier_discovery() -> None:
    config = ReachProxyConfig(
        grid=WorkspaceGridSpec(levels_mm=(20, 10, 5), x_slab_m=(1.0, 1.2)),
        required_consecutive_rounds=1,
        minimum_weighted_jaccard=1.0,
        maximum_new_volume_ratio=1.0,
        maximum_boundary_change_ratio=1.0,
        maximum_frontier_new_volume_ratio=0.0,
    )
    round_1 = ReachReplicaRound(
        round_id=1,
        replica_a=_replica("A", (41,), [[1.01, 0.0, 0.0]]),
        replica_b=_replica("B", (42,), [[1.01, 0.0, 0.0]]),
    )
    supplemental = CellKey(20, 55, 0, 0)

    result = ReachProxyBuilder(config).build(
        (round_1,), supplemental_supported_cells=(supplemental,)
    )

    assert supplemental in result.proxy_lower_cells
    assert result.status_by_cell[supplemental] is ReachStatus.EMPIRICAL_SUPPORTED
    assert result.replica_metrics[-1].frontier_new_volume_ratio == 0.0
    assert result.convergence_gate is True
