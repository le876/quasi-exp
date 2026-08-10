from __future__ import annotations

from dataclasses import replace

import pytest

from quasi_exp.teacher.protocol import (
    adaptive_family_radii,
    locate_radius_frontier,
    seal_family_splits,
    select_maximin_families,
    verify_family_split_seal,
)


def test_radius_frontier_refines_first_pass_fail_bracket_to_half_millimetre() -> None:
    calls: list[float] = []

    def evaluator(radius_mm: float) -> bool:
        calls.append(radius_mm)
        return radius_mm <= 101.0

    result = locate_radius_frontier(
        evaluator,
        coarse_anchors_mm=(75.0, 92.5, 100.0, 102.5, 103.0),
    )

    assert result.highest_pass_mm == 101.0
    assert result.first_fail_mm == 101.5
    assert result.resolution_mm == 0.5
    assert result.non_monotonic is False
    assert calls[:5] == [75.0, 92.5, 100.0, 102.5, 103.0]
    assert sorted(calls[5:]) == [100.5, 101.0, 101.5, 102.0]
    assert len(calls) == len(set(calls))


def test_radius_frontier_scans_full_anchor_range_after_non_monotonic_evidence() -> None:
    calls: list[float] = []

    def evaluator(radius_mm: float) -> bool:
        calls.append(radius_mm)
        return radius_mm <= 100.5 or radius_mm == 102.0

    result = locate_radius_frontier(
        evaluator,
        coarse_anchors_mm=(100.0, 101.0, 102.0, 103.0),
    )

    assert result.highest_pass_mm == 100.5
    assert result.first_fail_mm == 101.0
    assert result.non_monotonic is True
    assert result.post_failure_passes_mm == (102.0,)
    assert result.evaluated_radii_mm == (
        100.0,
        100.5,
        101.0,
        101.5,
        102.0,
        102.5,
        103.0,
    )


def test_maximin_family_selection_is_greedy_standardised_and_tie_deterministic() -> None:
    features = {
        "F5": (0.5, 0.5),
        "F3": (0.0, 1.0),
        "F1": (0.0, 0.0),
        "F4": (1.0, 1.0),
        "F2": (1.0, 0.0),
    }

    selected = select_maximin_families(
        features,
        count=2,
        seed_family_ids=("F1",),
    )

    assert selected == ("F4", "F2")


def test_five_adaptive_radii_floor_intermediate_points_to_half_millimetre_grid() -> None:
    assert adaptive_family_radii(102.5) == (75.0, 81.5, 88.5, 95.5, 102.5)

    with pytest.raises(ValueError, match="at least 92.5"):
        adaptive_family_radii(92.0)


def test_family_split_seal_is_canonical_and_keeps_virgin_families_test_only() -> None:
    seal = seal_family_splits(
        train_family_ids=("F2", "F1"),
        validation_family_ids=("F3",),
        virgin_test_family_ids=("F8", "F4"),
    )
    same_seal = seal_family_splits(
        train_family_ids=("F1", "F2"),
        validation_family_ids=("F3",),
        virgin_test_family_ids=("F4", "F8"),
    )

    assert seal.assignments == (
        ("F1", "train"),
        ("F2", "train"),
        ("F3", "validation"),
        ("F4", "virgin_test"),
        ("F8", "virgin_test"),
    )
    assert seal.training_family_ids == ("F1", "F2")
    assert seal.model_selection_family_ids == ("F3",)
    assert seal.virgin_family_ids == ("F4", "F8")
    assert seal.role_for("F4") == "virgin_test"
    assert seal.protocol_sha256 == same_seal.protocol_sha256
    assert verify_family_split_seal(seal) is True
    assert verify_family_split_seal(replace(seal, protocol_sha256="0" * 64)) is False
