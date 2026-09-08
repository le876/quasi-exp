from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.workspace_reach import CellKey, WorkspaceGridSpec
from quasi_exp.teacher.workspace_registry import (
    CellStratum,
    WorkspaceRegistryBuilder,
    WorkspaceRegistryPolicy,
)


BOUNDS = np.asarray([[-1.0, 1.0]] * 6, dtype=float)


def _row(
    sample_id: str,
    xyz: tuple[float, float, float],
    beta0: float,
    *,
    status: str = "empirical_supported",
    retention: bool = False,
    ill: bool = False,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "x_m": xyz[0],
        "y_m": xyz[1],
        "z_m": xyz[2],
        **{f"beta{index}_rad": beta0 + index * 0.001 for index in range(1, 7)},
        "reach_status": status,
        "retention": retention,
        "ill_conditioned": ill,
    }


def _builder(**policy: object) -> WorkspaceRegistryBuilder:
    return WorkspaceRegistryBuilder(
        WorkspaceGridSpec(levels_mm=(20, 10, 5), x_slab_m=(1.0, 1.2)),
        WorkspaceRegistryPolicy(beta_bounds_rad=BOUNDS, **policy),
    )


def test_builds_nested_cells_and_ten_mm_occupancy_from_supported_five_mm_children() -> None:
    rows = [
        _row("a", (1.001, 0.001, 0.001), -0.5),
        _row("b", (1.006, 0.001, 0.001), -0.2),
        _row("c", (1.001, 0.006, 0.001), 0.1),
        _row("d", (1.201, 0.001, 0.001), 0.3, status="unreachable_certified"),
    ]

    registry = _builder(measure_probe_count=2).build(pd.DataFrame(rows))

    parent = CellKey(10, 100, 0, 0)
    assert set(registry.cells_by_level) == {20, 10, 5}
    assert registry.cells_by_level[10][parent].sample_ids == ("a", "b", "c")
    assert registry.cells_by_level[10][parent].occupancy_fraction == pytest.approx(3 / 8)
    assert {CellStratum.BOUNDARY, CellStratum.TIP}.issubset(
        registry.cells_by_level[10][parent].strata
    )
    assert "d" not in registry.samples_by_id
    assert CellKey(5, 200, 0, 0) in registry.cells_by_level[5]


def test_explicit_overlapping_strata_preserve_empirical_evidence() -> None:
    rows = []
    # Make the central 10 mm cell interior by occupying every face-neighbor.
    offsets = (
        (0, 0, 0),
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    )
    for index, offset in enumerate(offsets):
        xyz = (
            1.105 + offset[0] * 0.01,
            0.005 + offset[1] * 0.01,
            0.005 + offset[2] * 0.01,
        )
        rows.append(_row(f"n{index}", xyz, -0.4 + index * 0.05))
    rows.extend(
        _row(
            f"center{index}",
            (1.105 + index * 0.0001, 0.005, 0.005),
            -0.1 + index * 0.1,
            retention=index == 0,
            ill=index == 1,
        )
        for index in range(3)
    )
    registry = _builder(low_density_max_samples=4).build(pd.DataFrame(rows))

    cell = registry.pilot_cells[CellKey(10, 110, 0, 0)]
    assert CellStratum.INTERIOR in cell.strata
    assert CellStratum.RETENTION in cell.strata
    assert CellStratum.ILL_CONDITIONED in cell.strata
    assert CellStratum.LOW_DENSITY in cell.strata
    assert CellStratum.BOUNDARY not in cell.strata


def test_pilot_is_deterministic_stratified_and_uses_only_real_samples() -> None:
    rows: list[dict[str, object]] = []
    for x_cell in (100, 101, 110, 111):
        for index in range(5):
            rows.append(
                _row(
                    f"x{x_cell}_{index}",
                    (x_cell / 100.0 + 0.001 + index * 0.0004, 0.001 + index * 0.001, 0.001),
                    -0.8 + index * 0.4,
                    retention=x_cell == 101 and index == 0,
                    ill=x_cell == 110 and index == 1,
                )
            )
    registry = _builder(
        measure_probe_count=3, max_seed_betas=4, selection_seed=7
    ).build(pd.DataFrame(rows))

    first = registry.select_pilot(4)
    second = registry.select_pilot(4)

    assert first == second
    assert len(first.cells) == 4
    assert {item.x_bin for item in first.cells} >= {0, 2}
    known = set(registry.samples_by_id)
    for item in first.cells:
        assert item.representative_sample_id in known
        assert len(item.measure_sample_ids) == 3
        assert set(item.measure_sample_ids) <= known
        assert item.representative_sample_id not in item.measure_sample_ids
        assert 1 <= len(item.seed_sample_ids) <= 4
        assert item.seed_beta_rad.shape == (len(item.seed_sample_ids), 6)
    assert len(first.cells) == min(4, first.eligible_cell_count)


def test_six_face_task_graph_and_budget_cap_are_stable() -> None:
    rows = [
        _row(f"a{index}", (1.001 + index * 0.0005, 0.001, 0.001), -0.5 + index * 0.2)
        for index in range(4)
    ] + [
        _row(f"b{index}", (1.011 + index * 0.0005, 0.001, 0.001), -0.5 + index * 0.2)
        for index in range(4)
    ]
    registry = _builder(measure_probe_count=2, selection_seed=1).build(pd.DataFrame(rows))

    selection = registry.select_pilot(99)

    assert len(selection.cells) == 2
    assert selection.task_edges == ((CellKey(10, 100, 0, 0), CellKey(10, 101, 0, 0)),)
    assert registry.select_pilot(1).task_edges == ()


def test_rejects_capability_beta_outside_registered_bounds() -> None:
    row = _row("bad", (1.001, 0.001, 0.001), 2.0)
    with pytest.raises(ValueError, match="registered beta bounds"):
        _builder().build(pd.DataFrame([row]))
