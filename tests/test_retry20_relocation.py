import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.retry20_relocation import (
    single_macroblock, hash_order, SpacingIndex, phase_audit, geometry_gate,
    recover_rows, select_replacements,
)
from quasi_exp.teacher.retry12_symmetry import XYZ_COLUMNS, BETA_COLUMNS


@pytest.mark.parametrize("lower,upper,expected", [
    ([30, -40, 0], [40, -30, 10], True),
    ([-10, -10, -10], [0, 0, 0], True),
    ([35, 0, 0], [45, 10, 10], False),
    ([-45, 0, 0], [-35, 10, 10], False),
])
def test_half_open_macroblock(lower, upper, expected):
    assert single_macroblock(lower, upper) == expected


def test_hash_order_is_input_order_independent():
    ids = [f"box10:{i}:0:0" for i in range(20)]
    assert hash_order(ids, 7) == hash_order(ids[::-1], 7)
    assert hash_order(ids, 7) != sorted(ids)


def test_spacing_removes_old_coordinate_and_keeps_other_rows():
    s = SpacingIndex([[0, 0, 0], [2, 0, 0]], 1)
    assert s.allowed([.2, 0, 0], 0)
    assert not s.allowed([1.5, 0, 0], 0)
    s.replace(0, [-2, 0, 0])
    assert s.allowed([0, 0, 0], 1)


def test_joint_entropy_detects_diagonal_phase_distribution():
    phase = np.linspace(.001, .999, 1000)
    audit = phase_audit(np.tile(phase[:, None], (1, 3)) * .01)
    assert audit["x"]["entropy"] > .99
    assert audit["joint_entropy_4cubed"] < .34


def test_count_and_phase_cannot_override_coverage_failure():
    base = {"volume_coverage": .985, "minimum_axial_slice_coverage": .92}
    proposed = {"volume_coverage": .96, "minimum_axial_slice_coverage": .92,
                "service_distance_volume_weighted_p95_mm": 10.000000000000009}
    phase = {"x": {"entropy": .95, "max_bin_fraction": .15}, "center_plane_fraction": .1}
    gate = geometry_gate(base, proposed, phase, 24000)
    assert not gate["passed"]
    assert gate["checks"]["service_p95"]
    assert gate["failed_checks"] == ["volume_coverage"]
    proposed["volume_coverage"] = .98
    assert geometry_gate(base, proposed, phase, 24000)["passed"]
    assert not geometry_gate(base, proposed, phase, 23999)["passed"]


def fixture_rows():
    labels = pd.DataFrame([
        {"target_id": "zero", "x_m": 1.215498, "y_m": 0., "z_m": 0., **dict.fromkeys(BETA_COLUMNS, 0.)},
        {"target_id": "root", "x_m": 1.21, "y_m": 0., "z_m": 0., **dict.fromkeys(BETA_COLUMNS, .001)},
        {"target_id": "cell", "x_m": 1.095, "y_m": .005, "z_m": .005, **dict.fromkeys(BETA_COLUMNS, .1)},
    ])
    targets = labels.copy()
    targets["target_role"] = ["exact_zero", "primary_cell_service", "primary_cell_service"]
    targets["cell_id"] = [None, "box5:242:0:0", "box10:109:0:0"]
    for axis in "xyz":
        targets[f"{axis}_min_mm"] = [np.nan, 1210 if axis == "x" else 0, 1090 if axis == "x" else 0]
        targets[f"{axis}_max_mm"] = targets[f"{axis}_min_mm"] + [np.nan, 5, 10]
    targets["cell_size_mm"] = [np.nan, 5, 10]
    unified = labels.copy()
    unified["split_role"] = "train"
    unified["sample_weight"] = 1.
    unified["macroblock_id"] = ["smacro:30:0:0", "smacro:30:0:0", "smacro:27:0:0"]
    anchors = pd.DataFrame({"target_id": ["zero"], "anchor_class": ["hard"]})
    edges = pd.DataFrame({"left_target_id": ["zero"], "right_target_id": ["root"], "teacher_edge_legal": [True]})
    rows = recover_rows(labels, targets, unified, anchors, edges, 1.215498)
    return rows, unified, (labels, targets, anchors, edges)


def test_zero_and_certified_root_are_not_relocated():
    rows, _, _ = fixture_rows()
    assert rows.relocation_eligible.tolist() == [False, False, True]
    assert rows.relocation_exclusion_reason.tolist() == ["exact_zero", "hard_teacher_zero_attachment", ""]


def test_missing_provenance_or_zero_identity_rejected():
    rows, unified, (labels, targets, anchors, edges) = fixture_rows()
    with pytest.raises(ValueError, match="role or final split"):
        recover_rows(labels, targets.iloc[:2], unified, anchors, edges, 1.215498)
    labels.loc[0, BETA_COLUMNS[0]] = .01
    with pytest.raises(ValueError, match="zero beta"):
        recover_rows(labels, targets, unified, anchors, edges, 1.215498)


def test_actual_phase_reserved_coordinate_and_spacing_filters():
    rows, unified, _ = fixture_rows()
    proposals = pd.DataFrame([
        {"proposal_id": name, "proposal_pool": "a", "source_cell_id": "box10:109:0:0",
         "x_m": 1.090 + phase*.01, "y_m": .005, "z_m": .005,
         "phase_x": phase, "phase_y": .5, "phase_z": .5,
         "same_cell_cross_pool_supported": True, "u_mm": 120.,
         "macroblock_id": "smacro:27:0:0", "split_role": "train"}
        for name, phase in [("boundary", .01), ("reserved", .5), ("valid", .8)]
    ])
    panels = proposals[proposals.proposal_id.eq("reserved")]
    _, selected = select_replacements(rows, proposals, panels, unified, seed=1)
    assert selected.proposal_id.tolist() == ["valid"]
    assert selected.source_cell_probe_served.all()
    assert selected.target_id.tolist() == ["cell"]
