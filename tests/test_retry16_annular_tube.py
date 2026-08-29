from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS
from quasi_exp.teacher.retry16_annular_tube import (
    AnnularPolicy,
    build_annular_target_registry,
    build_nested_annular_profiles,
    intersect_support,
    root_connector_voxels,
)


def _support(*, inner_by_u: list[int], outer_by_u: list[int]) -> pd.DataFrame:
    rows = []
    for u_index, (inner, outer) in enumerate(zip(inner_by_u, outer_by_u, strict=True)):
        for rho_index in range(inner, outer + 1):
            for sector_index in range(16):
                rows.append(
                    {
                        "u_index": u_index,
                        "rho_index": rho_index,
                        "sector_index": sector_index,
                        "robust_supported": True,
                    }
                )
    return pd.DataFrame(rows)


def _profiles() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # The final scale sees a wider annulus.  The comparison scale is nested and
    # deliberately lacks the first/last axial bins.
    final = _support(inner_by_u=[1] * 60, outer_by_u=[15] * 60)
    comparison = _support(
        inner_by_u=[20, *([2] * 58), 20],
        outer_by_u=[20, *([14] * 58), 20],
    )
    _raw_expanded, _raw_core, expanded, core = build_nested_annular_profiles(
        final,
        comparison,
        policy=AnnularPolicy(inner_margin_mm=0.0, outer_margin_mm=0.0),
    )
    return final, expanded, core


def test_nested_annulus_excludes_hollow_core_and_freezes_stable_intersection() -> None:
    final, expanded, core = _profiles()
    assert len(expanded) == 60
    assert expanded["inner_radius_mm"].min() == 10.0
    assert expanded["outer_radius_mm"].max() == 160.0
    assert core["u_index"].min() == 1
    assert core["u_index"].max() == 58
    assert core["inner_radius_mm"].min() == 20.0
    assert core["outer_radius_mm"].max() == 150.0
    intersection = intersect_support(final, final)
    assert len(intersection) == len(final)


def test_root_connector_is_explicit_and_never_annular_coverage() -> None:
    final, expanded, _core = _profiles()
    connector = root_connector_voxels(
        final,
        expanded,
        policy=AnnularPolicy(
            inner_margin_mm=0.0,
            outer_margin_mm=0.0,
            root_link_maximum_mm=25.0,
        ),
    )
    assert len(connector) >= 1
    assert connector["connector_ordinal"].tolist() == list(range(len(connector)))
    assert not connector["annular_coverage_eligible"].any()
    first_distance = np.linalg.norm(
        connector.loc[0, ["u_mm", "y_mm", "z_mm"]].to_numpy(float)
    )
    assert first_distance <= 25.0


def test_registry_has_zero_connector_annulus_and_nine_large_circles_without_axis_core() -> None:
    final, expanded, core = _profiles()
    policy = AnnularPolicy(
        inner_margin_mm=0.0,
        outer_margin_mm=0.0,
        root_link_maximum_mm=25.0,
    )
    connector = root_connector_voxels(final, expanded, policy=policy)
    proposals = np.asarray(
        [
            [1.2154980000000004 - (u + 0.5) * 0.01, rho * 0.01, 0.0]
            for u in range(60)
            for rho in range(1, 17)
        ],
        dtype=float,
    )
    targets, edges, circles, summary = build_annular_target_registry(
        expanded,
        core,
        connector,
        proposals,
        target_budget=5000,
        maximum_circle_step_mm=10.0,
        policy=policy,
    )
    assert summary["exact_zero_count"] == 1
    assert summary["axis_core_target_count"] == 0
    assert summary["connector_target_count"] >= 1
    assert summary["registered_circle_count"] == 9
    assert summary["large_circle_count"] >= 3
    assert summary["outside_expanded_annulus_count"] == 0
    assert not any(name in targets for name in BETA_COLUMNS)
    assert set(edges["edge_type"]) == {"circle", "root_connector"}
    assert circles.groupby("circle_id")["held_out"].all().all()
    assert not targets.loc[
        targets["target_role"].isin(["exact_zero", "root_connector"]),
        "annular_coverage_eligible",
    ].any()
