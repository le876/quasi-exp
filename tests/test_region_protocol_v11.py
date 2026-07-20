from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.region import EllipseFamilySpec
from quasi_exp.teacher.region_protocol import (
    FamilyCatalog,
    generate_family_catalog,
    nested_family_sample_indices,
    promote_complete_families,
    require_boolean_gate_tree,
)


def _baseline() -> EllipseFamilySpec:
    return EllipseFamilySpec(
        family_id="anchor",
        center_m=np.asarray([0.5, 0.0, 0.1]),
        major_direction=np.asarray([0.0, 1.0, 0.0]),
        minor_direction=np.asarray([0.0, 0.0, 1.0]),
        major_semiaxis_m=0.5,
        minor_semiaxis_m=0.17,
    )


def test_family_catalog_is_frozen_deterministic_and_has_role_specific_reserves() -> None:
    catalog = generate_family_catalog(_baseline(), seed=20260721)
    repeated = generate_family_catalog(_baseline(), seed=20260721)

    assert isinstance(catalog, FamilyCatalog)
    assert catalog.fingerprint == repeated.fingerprint
    pd.testing.assert_frame_equal(catalog.frame, repeated.frame)
    assert len(catalog.frame) == 72
    quota = catalog.frame.groupby(["role", "is_primary"]).size().to_dict()
    assert quota == {
        ("train", False): 28,
        ("train", True): 14,
        ("validation", False): 10,
        ("validation", True): 5,
        ("virgin_test", False): 10,
        ("virgin_test", True): 5,
    }
    assert catalog.frame["family_id"].is_unique
    assert catalog.frame["major_semiaxis_m"].between(0.46, 0.50).all()
    assert catalog.frame["axis_ratio"].between(0.31, 0.36).all()


def test_sealed_test_ids_are_not_available_to_model_selection() -> None:
    catalog = generate_family_catalog(_baseline(), seed=20260721)

    model_ids = set(catalog.authorized_family_ids("model_selection"))
    test_ids = set(catalog.frame.loc[catalog.frame["role"].eq("virgin_test"), "family_id"])

    assert model_ids
    assert model_ids.isdisjoint(test_ids)
    with pytest.raises(PermissionError, match="sealed"):
        catalog.authorized_family_ids("sealed_evaluation")
    assert set(
        catalog.authorized_family_ids(
            "sealed_evaluation", unseal_token=catalog.seal_token
        )
    ) == set(catalog.primary_ids("virgin_test"))


def test_frozen_relative_family_design_can_be_applied_to_selected_anchor() -> None:
    catalog = generate_family_catalog(_baseline(), seed=20260721)
    shifted_anchor = EllipseFamilySpec(
        family_id="selected-anchor",
        center_m=_baseline().center_m + np.asarray([0.02, -0.01, 0.03]),
        major_direction=_baseline().major_direction,
        minor_direction=_baseline().minor_direction,
        major_semiaxis_m=0.5,
        minor_semiaxis_m=0.17,
    )

    applied = catalog.reanchor(shifted_anchor)

    assert applied.frame["family_id"].tolist() == catalog.frame["family_id"].tolist()
    assert applied.frame[["role", "is_primary", "reserve_rank"]].equals(
        catalog.frame[["role", "is_primary", "reserve_rank"]]
    )
    delta = applied.frame[["center_x_m", "center_y_m", "center_z_m"]].to_numpy() - catalog.frame[
        ["center_x_m", "center_y_m", "center_z_m"]
    ].to_numpy()
    assert np.allclose(delta, shifted_anchor.center_m - _baseline().center_m)
    assert applied.fingerprint != catalog.fingerprint


def test_family_promotion_is_all_or_nothing_and_uses_frozen_reserve_order() -> None:
    catalog = generate_family_catalog(_baseline(), seed=20260721)
    quality = catalog.frame[["family_id", "role", "is_primary", "reserve_rank"]].copy()
    quality["complete_gate_pass"] = True
    rejected = catalog.primary_ids("train")[0]
    quality.loc[quality["family_id"].eq(rejected), "complete_gate_pass"] = False

    selection = promote_complete_families(catalog, quality)

    assert selection.groupby("role").size().to_dict() == {
        "train": 14,
        "validation": 5,
        "virgin_test": 5,
    }
    assert rejected not in set(selection["family_id"])
    promoted = selection[(selection["role"].eq("train")) & (~selection["is_primary"])]
    assert promoted["reserve_rank"].tolist() == [0]


def test_nested_sample_indices_include_every_training_family_and_are_nested() -> None:
    rows = []
    for family_index in range(14):
        for phase in range(24):
            for cross_section in range(9):
                rows.append(
                    {
                        "family_id": f"F{family_index:02d}",
                        "phase_idx": phase,
                        "cross_section_idx": cross_section,
                        "normalized_radial_offset": 1.0 if cross_section else 0.0,
                        "normalized_plane_offset": 0.0,
                    }
                )
    frame = pd.DataFrame(rows)

    subsets = nested_family_sample_indices(
        frame,
        sizes=(200, 500, 1000, 2000),
        seed=20260810,
    )

    previous: set[int] = set()
    for size, indices in subsets.items():
        assert len(indices) == size
        assert previous.issubset(set(indices))
        assert frame.loc[indices, "family_id"].nunique() == 14
        previous = set(indices)


def test_gate_tree_rejects_numeric_truth_values_and_missing_raw_checks(tmp_path: Path) -> None:
    assert require_boolean_gate_tree(
        {"gate_pass": True, "checks": {"residual": True, "margin": True}}
    )
    with pytest.raises(TypeError, match="bool"):
        require_boolean_gate_tree(
            {"gate_pass": 1, "checks": {"residual": True, "margin": True}}
        )
    with pytest.raises(ValueError, match="checks"):
        require_boolean_gate_tree({"gate_pass": True})
