from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.retry17_continuity_fill import (
    continuous_annular_sobol,
    density_normalized_select_t2,
    geometric_graph,
    graph_lcc_fraction,
    graph_stretch,
    local_directional_coverage,
    progressive_farthest_fill,
    teacher_edge_metrics,
)


ZERO_X = 1.2154980000000004


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "u_center_mm": [0.0, 100.0, 200.0],
            "inner_radius_mm": [10.0, 10.0, 10.0],
            "outer_radius_mm": [200.0, 200.0, 200.0],
        }
    )


def _point(target_id: str, u: float, rho: float, phi: float) -> dict[str, float | str | bool]:
    return {
        "target_id": target_id,
        "x_m": ZERO_X - u / 1000.0,
        "y_m": rho * np.cos(phi) / 1000.0,
        "z_m": rho * np.sin(phi) / 1000.0,
        "annular_coverage_eligible": True,
    }


def test_continuous_sobol_is_volume_correct_and_has_no_fixed_u_layers() -> None:
    pool = continuous_annular_sobol(
        _profile(), power=12, seed=20260911, zero_x_m=ZERO_X, pool_id="test"
    )
    assert pool["u_mm"].nunique() == len(pool)
    assert pool["phi_rad"].between(0.0, 0.5 * np.pi).all()
    expected_mean_rho2 = 0.5 * (10.0**2 + 200.0**2)
    assert abs(float(np.mean(np.square(pool["rho_mm"]))) - expected_mean_rho2) < 30.0


def test_progressive_fill_respects_minimum_separation_and_is_deterministic() -> None:
    pool = continuous_annular_sobol(
        _profile(), power=9, seed=20260911, zero_x_m=ZERO_X, pool_id="fill"
    )
    first, audit = progressive_farthest_fill(
        pool,
        np.asarray([[ZERO_X, 0.0, 0.0]]),
        maximum_new_points=80,
        minimum_separation_mm=15.0,
        batch_size=20,
        target_fill_mm=25.0,
    )
    second, _ = progressive_farthest_fill(
        pool,
        np.asarray([[ZERO_X, 0.0, 0.0]]),
        maximum_new_points=80,
        minimum_separation_mm=15.0,
        batch_size=20,
        target_fill_mm=25.0,
    )
    assert first["target_id"].tolist() == second["target_id"].tolist()
    distance = np.linalg.norm(
        first.loc[:, XYZ_COLUMNS].to_numpy(float)[:, None, :]
        - first.loc[:, XYZ_COLUMNS].to_numpy(float)[None, :, :],
        axis=2,
    )
    distance[distance == 0.0] = np.inf
    assert float(np.min(distance) * 1000.0) >= 15.0 - 1e-9
    assert audit["selected_total"].tolist() == [20, 40, 60, 80]


def test_directional_coverage_observes_all_six_cylindrical_directions() -> None:
    rows = [_point("center", 100.0, 100.0, 0.7)]
    rows += [
        _point("u-", 90.0, 100.0, 0.7),
        _point("u+", 110.0, 100.0, 0.7),
        _point("r-", 100.0, 90.0, 0.7),
        _point("r+", 100.0, 110.0, 0.7),
        _point("p-", 100.0, 100.0, 0.6),
        _point("p+", 100.0, 100.0, 0.8),
    ]
    coverage, audit = local_directional_coverage(
        pd.DataFrame(rows),
        _profile(),
        zero_x_m=ZERO_X,
        neighbourhood_radius_mm=40.0,
        projection_minimum_mm=5.0,
        cosine_minimum=0.5,
    )
    center = audit[audit["target_id"].eq("center")].iloc[0]
    assert bool(center["all_six_supported"])
    assert coverage > 0.0


def test_geometric_graph_lcc_and_stretch_are_independent_of_teacher_edges() -> None:
    targets = pd.DataFrame([_point(str(i), 50.0 + 10.0 * i, 100.0, 0.7) for i in range(8)])
    edges = geometric_graph(targets, k=2, maximum_distance_mm=15.0)
    assert graph_lcc_fraction(targets, edges) == 1.0
    metrics, pairs = graph_stretch(targets, edges, pair_count=20, seed=20260913)
    assert len(pairs) > 0
    assert metrics["finite_fraction"] == 1.0
    assert metrics["p95"] <= 1.000001


def test_density_normalized_teacher_never_overwrites_locked_single_candidate() -> None:
    targets = pd.DataFrame([_point("old", 80.0, 100.0, 0.7), _point("new", 90.0, 100.0, 0.7)])
    edges = pd.DataFrame(
        [{"left_target_id": "old", "right_target_id": "new", "distance_mm": 10.0}]
    )
    rows = []
    for target, candidate, value in (("old", "locked", 0.2), ("new", "near", 0.21), ("new", "zero", 0.0)):
        row = {
            "target_id": target,
            "candidate_id": candidate,
            "solver_success": True,
            "bounds_pass": True,
            "fk_residual_mm": 0.0,
            "min_margin_deg": 1.0,
        }
        row.update(dict(zip(BETA_COLUMNS, [value, 0, 0, 0, 0, 0], strict=True)))
        rows.append(row)
    labels = density_normalized_select_t2(
        pd.DataFrame(rows),
        edges,
        pairwise_lambda=8.0,
        sigma_mm=25.0,
        locked_target_ids=["old"],
    )
    assert labels.set_index("target_id").loc["old", "candidate_id"] == "locked"


def test_teacher_edge_metrics_reads_tuple_beta_columns_from_indexed_rows() -> None:
    rows = []
    for target, value in (("left", 0.0), ("right", 0.01)):
        row = {"target_id": target}
        row.update(dict(zip(BETA_COLUMNS, [value, 0, 0, 0, 0, 0], strict=True)))
        rows.append(row)
    edges = pd.DataFrame(
        [{"left_target_id": "left", "right_target_id": "right", "distance_mm": 10.0}]
    )
    metrics = teacher_edge_metrics(pd.DataFrame(rows), edges, weights=(4, 4, 2, 2, 1, 1))
    assert metrics["edge_count"] == 1
    assert metrics["weighted_p95_deg"] > 0.0
