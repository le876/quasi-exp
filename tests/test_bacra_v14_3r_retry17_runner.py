from __future__ import annotations

from pathlib import Path

from scripts.analysis.run_bacra_v14_3r_retry17_continuity import (
    EXPERIMENT_ID,
    STAGE_DIRS,
    _effective_seed_budget,
    _merge_label_metadata,
    _strip_target_metadata_from_labels,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_retry17_config_locks_independent_seeds_caps_and_false_claims() -> None:
    config = load_config(ROOT / "configs/bacra_v14_3r_retry17_omega600_annular_continuity.yaml")
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["sampling"]["phase_a_quotient_cap"] == 7000
    assert config["sampling"]["phase_b_quotient_cap"] == 14000
    assert [config["sampling"][key] for key in ("evaluation_probe_seed", "target_pool_seed", "trajectory_center_seed", "graph_stretch_seed", "spatial_split_seed", "student_seed")] == list(range(20260910, 20260916))
    assert config["claims"]["diagnostic_only"]
    assert not any(value for key, value in config["claims"].items() if key != "diagnostic_only")


def test_retry17_stage_order_keeps_shapes_after_dataset_student_lock() -> None:
    names = list(STAGE_DIRS)
    assert names.index("dataset_student_lock") < names.index("heldout_shapes") < names.index("trajectory_evaluation")
    assert list(STAGE_DIRS.values())[-1] == "12_summary"


def test_retry17_protocol_and_launcher_are_bound_to_one_experiment() -> None:
    config = load_config(ROOT / "configs/bacra_v14_3r_retry17_omega600_annular_continuity.yaml")
    assert (ROOT / config["sources"]["governing_protocol"]).is_file()
    launcher = (ROOT / "scripts/pipelines/run_bacra_v14_3r_retry17_continuity.sh").read_text(encoding="utf-8")
    assert "BACRA_RETRY17_CONTINUITY_BINDING_SHA" in launcher
    assert "run_bacra_v14_3r_retry17_continuity.py" in launcher
    assert "CUDA_VISIBLE_DEVICES=-1" in launcher


def test_retry17_smoke_reuses_a_legal_candidate_solver_budget() -> None:
    assert _effective_seed_budget(16, smoke=True) == 8
    assert _effective_seed_budget(16, smoke=False) == 16


def test_retry17_target_registry_owns_metadata_when_locked_labels_overlap() -> None:
    import pandas as pd

    labels = pd.DataFrame([{"target_id": "x", "target_role": "stale", "x_m": 9.0, "beta1_rad": 0.0}])
    targets = pd.DataFrame([{"target_id": "x", "target_role": "root_connector", "x_m": 1.0, "y_m": 0.0, "z_m": 0.0}])
    merged = _merge_label_metadata(labels, targets)
    assert merged.loc[0, "target_role"] == "root_connector"
    assert merged.loc[0, "x_m"] == 1.0
    assert not any(column.endswith(("_x", "_y")) for column in merged.columns)


def test_retry17_symmetry_labels_do_not_reintroduce_target_metadata() -> None:
    import pandas as pd

    labels = pd.DataFrame([{"target_id": "x", "target_role": "stale", "x_m": 9.0, "beta1_rad": 0.1, "candidate_id": "c"}])
    targets = pd.DataFrame([{"target_id": "x", "target_role": "exact_zero", "x_m": 1.0, "y_m": 0.0, "z_m": 0.0}])
    stripped = _strip_target_metadata_from_labels(labels, targets)
    assert set(stripped) == {"target_id", "beta1_rad", "candidate_id"}
