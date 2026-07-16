from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    analysis_dir = REPO_ROOT / "scripts" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    path = analysis_dir / "true_ellipse_radial_bundle_v6_utils.py"
    spec = importlib.util.spec_from_file_location("true_ellipse_radial_bundle_v6_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shared_json_io_serializes_numpy_values_and_paths(tmp_path: Path) -> None:
    mod = _load_module()
    output = tmp_path / "nested" / "report.json"

    mod.write_json(
        output,
        {
            "count": np.int64(3),
            "score": np.float64(1.25),
            "passed": np.bool_(True),
            "values": np.asarray([1, 2]),
            "path": tmp_path,
        },
    )

    assert mod.read_json(output) == {
        "count": 3,
        "score": 1.25,
        "passed": True,
        "values": [1, 2],
        "path": str(tmp_path),
    }


def test_v6_target_is_rank2_and_preserves_fixed_family_phase() -> None:
    mod = _load_module()
    family = mod.FamilySpec(
        family_id="fixed-family",
        center_x_m=1.112351,
        center_y_m=0.092529,
        center_z_m=-0.156536,
        phase_y_rad=math.radians(208.477),
        phase_z_rad=math.radians(341.493),
    )

    targets = mod.generate_radius_targets(family, radius_mm=100.0, n_points=360)
    report = mod.validate_target_geometry(targets, family=family, radius_mm=100.0)

    assert report["target_geometry_gate_pass"] is True
    assert report["rank2_gate_pass"] is True
    assert report["phase_fields_preserved"] is True
    assert np.isclose(targets["phase_y_rad"], family.phase_y_rad).all()
    assert np.isclose(targets["phase_z_rad"], family.phase_z_rad).all()
    assert np.isclose(targets["amp_xy_mm"], 100.0).all()
    assert np.isclose(targets["amp_z_mm"], 150.0).all()


def _path(mod, *, radius_mm: float, family_id: str = "fixed-family", offset_deg: float = 0.0) -> pd.DataFrame:
    angle_idx = np.arange(4, dtype=np.int64)
    values = np.deg2rad(angle_idx[:, None] + np.arange(6, dtype=float)[None, :] + float(offset_deg))
    frame = pd.DataFrame(
        {
            "angle_idx": angle_idx,
            "angle_rad": angle_idx * (2.0 * math.pi / 4.0),
            "family_id": family_id,
            "radius_mm": float(radius_mm),
        }
    )
    for column_idx, column in enumerate(mod.atlas.BETA_COLS):
        frame[column] = values[:, column_idx]
    return frame


def test_parent_copy_predictor_uses_the_full_angle_aligned_parent_path() -> None:
    mod = _load_module()
    parent = _path(mod, radius_mm=80.0).iloc[[2, 0, 3, 1]].reset_index(drop=True)

    predictor, report = mod.build_radial_predictor(
        parent,
        parent_radius_mm=80.0,
        target_radius_mm=81.0,
        family_id="fixed-family",
    )

    expected = parent.sort_values("angle_idx")[mod.atlas.BETA_COLS].to_numpy(dtype=float)
    assert report["radial_predictor_type"] == "parent_copy"
    assert report["full_parent_path_used"] is True
    assert predictor["angle_idx"].tolist() == [0, 1, 2, 3]
    assert np.allclose(predictor[mod.atlas.BETA_COLS], expected)
    assert not np.allclose(predictor[mod.atlas.BETA_COLS], expected[0])


def test_radial_secant_predictor_is_computed_at_every_matching_angle() -> None:
    mod = _load_module()
    previous = _path(mod, radius_mm=75.0, offset_deg=-0.5)
    parent = _path(mod, radius_mm=80.0, offset_deg=0.5)

    predictor, report = mod.build_radial_predictor(
        parent,
        parent_radius_mm=80.0,
        target_radius_mm=85.0,
        previous_parent=previous,
        previous_parent_radius_mm=75.0,
        family_id="fixed-family",
        bounds=np.deg2rad(np.asarray([[-30.0, 30.0]] * 6)),
    )

    parent_beta = parent[mod.atlas.BETA_COLS].to_numpy(dtype=float)
    previous_beta = previous[mod.atlas.BETA_COLS].to_numpy(dtype=float)
    assert report["radial_predictor_type"] == "radial_secant"
    assert np.allclose(predictor[mod.atlas.BETA_COLS], parent_beta + (parent_beta - previous_beta))
    assert predictor["parent_radius_mm"].eq(80.0).all()
    assert predictor["previous_parent_radius_mm"].eq(75.0).all()


def test_cut_rotation_round_trip_preserves_angle_beta_and_target_alignment() -> None:
    mod = _load_module()
    frame = _path(mod, radius_mm=80.0)
    frame["x_target_m"] = frame["angle_idx"] + 1.0
    frame["y_target_m"] = frame["angle_idx"] + 2.0
    frame["z_target_m"] = frame["angle_idx"] + 3.0

    rotated = mod.rotate_for_cut(frame, cut_idx=2)
    restored = mod.restore_angle_order(rotated)

    assert rotated["angle_idx"].tolist() == [2, 3, 0, 1]
    assert rotated["solver_order_idx"].tolist() == [0, 1, 2, 3]
    pd.testing.assert_frame_equal(
        restored.drop(columns="solver_order_idx").reset_index(drop=True),
        frame.sort_values("angle_idx").reset_index(drop=True),
    )


def test_cyclic_deltas_include_the_last_to_first_seam_and_anchor_never_drops_to_zero() -> None:
    mod = _load_module()
    beta = np.zeros((4, 6), dtype=float)
    beta[-1, :] = np.deg2rad(2.0)

    deltas = mod.cyclic_beta_delta_rms_deg(beta)
    schedules = {name: mod.anchor_stages(name) for name in mod.ANCHOR_SCHEDULES}

    assert len(deltas) == len(beta)
    assert np.isclose(deltas[-1], 2.0)
    assert set(schedules) == {"conservative", "balanced", "loose"}
    assert [stage["lambda_anchor"] for stage in schedules["conservative"]] == [1.0, 0.3, 0.1]
    assert all(float(stage["lambda_anchor"]) > 0.0 for stages in schedules.values() for stage in stages)


def test_cut_invariance_requires_every_registered_run_and_pairwise_p95_within_one_degree() -> None:
    mod = _load_module()
    baseline = _path(mod, radius_mm=81.0)
    paths = {
        (cut, predictor): mod.rotate_for_cut(baseline, cut_idx=cut)
        for cut in (0, 1, 2, 3)
        for predictor in ("parent_copy", "radial_secant")
    }

    passing = mod.cut_invariance_report(
        paths,
        required_cuts=(0, 1, 2, 3),
        required_predictors=("parent_copy", "radial_secant"),
    )
    drifted = {key: value.copy() for key, value in paths.items()}
    for column in mod.atlas.BETA_COLS:
        drifted[(3, "radial_secant")][column] += np.deg2rad(1.25)
    failing = mod.cut_invariance_report(
        drifted,
        required_cuts=(0, 1, 2, 3),
        required_predictors=("parent_copy", "radial_secant"),
    )

    assert passing["cut_invariance_gate_pass"] is True
    assert passing["required_run_count"] == 8
    assert failing["cut_invariance_gate_pass"] is False
    assert failing["pairwise_branch_diff_p95_max_deg"] > 1.0


def test_adaptive_radius_walk_retries_half_then_quarter_steps_without_changing_family() -> None:
    mod = _load_module()
    initial = _path(mod, radius_mm=80.0)
    calls: list[float] = []

    def solve_step(parent: pd.DataFrame, parent_radius_mm: float, target_radius_mm: float):
        del parent, parent_radius_mm
        calls.append(target_radius_mm)
        success = not np.isclose(target_radius_mm, 81.0) or calls.count(81.0) > 1
        return _path(mod, radius_mm=target_radius_mm), {"radial_bundle_gate_pass": success}

    paths, report = mod.adaptive_radius_walk(
        initial,
        start_radius_mm=80.0,
        checkpoints_mm=(81.0, 82.0),
        family_id="fixed-family",
        solve_step=solve_step,
        base_step_mm=1.0,
        retry_steps_mm=(0.5, 0.25),
    )

    assert calls[:3] == [81.0, 80.5, 81.0]
    assert report["radial_walk_gate_pass"] is True
    assert report["last_pass_radius_mm"] == 82.0
    assert set(paths) >= {80.0, 80.5, 81.0, 82.0}
    assert all(path["family_id"].eq("fixed-family").all() for path in paths.values())


def test_adaptive_radius_walk_rejects_cross_family_branch_merging() -> None:
    mod = _load_module()

    def wrong_family(_parent: pd.DataFrame, _parent_radius_mm: float, target_radius_mm: float):
        return _path(mod, radius_mm=target_radius_mm, family_id="other-family"), {
            "radial_bundle_gate_pass": True
        }

    with pytest.raises(ValueError, match="fixed-family invariant"):
        mod.adaptive_radius_walk(
            _path(mod, radius_mm=80.0),
            start_radius_mm=80.0,
            checkpoints_mm=(81.0,),
            family_id="fixed-family",
            solve_step=wrong_family,
        )


def test_whole_radius_split_and_support_pool_have_no_heldout_leakage() -> None:
    mod = _load_module()
    rows = []
    for radius_mm in mod.FORMAL_RADII_MM:
        for angle_idx in range(2):
            for offset_id in ("center", "edge"):
                rows.append(
                    {
                        "family_id": "fixed-family",
                        "trajectory_id": f"fixed-family@{radius_mm:g}",
                        "radius_mm": radius_mm,
                        "angle_idx": angle_idx,
                        "tube_offset_id": offset_id,
                        "is_centerline": offset_id == "center",
                        "x_target_m": radius_mm / 1000.0,
                        "y_target_m": float(angle_idx),
                        "z_target_m": 0.0,
                    }
                )
    dataset = pd.DataFrame(rows)

    assigned, report = mod.assign_whole_radius_splits(dataset)
    support_pool = mod.training_only_support_pool(assigned)

    assert report["whole_radius_split_gate_pass"] is True
    assert set(assigned.loc[assigned["split"].eq("validation"), "radius_mm"]) == {92.5}
    assert set(assigned.loc[assigned["split"].eq("test"), "radius_mm"]) == {100.0}
    assert 92.5 not in set(support_pool["radius_mm"])
    assert 100.0 not in set(support_pool["radius_mm"])
    assert set(support_pool["split"]) == {"train"}
    assert not support_pool["is_centerline"].any()


def test_training_only_support_fails_closed_without_a_boolean_centerline_flag() -> None:
    mod = _load_module()
    valid = pd.DataFrame(
        {
            "split": ["train", "train"],
            "radius_mm": [75.0, 75.0],
            "is_centerline": [False, True],
        }
    )

    with pytest.raises(ValueError, match="is_centerline"):
        mod.training_only_support_pool(valid.drop(columns="is_centerline"))
    with pytest.raises(ValueError, match="boolean"):
        mod.training_only_support_pool(valid.assign(is_centerline=["false", "true"]))


def test_radial_cache_fingerprint_binds_parent_content_family_radius_schedule_config_and_version(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    parent = tmp_path / "parent.parquet"
    config = tmp_path / "robot.yaml"
    parent.write_bytes(b"parent-v1")
    config.write_text("robot: v1\n", encoding="utf-8")

    base = mod.radial_task_fingerprint(
        family_id="fixed-family",
        target_radius_mm=81.0,
        parent_paths=(parent,),
        anchor_schedule="balanced",
        robot_config_path=config,
        strategy_version=7,
    )
    changed_family = mod.radial_task_fingerprint(
        family_id="other-family",
        target_radius_mm=81.0,
        parent_paths=(parent,),
        anchor_schedule="balanced",
        robot_config_path=config,
        strategy_version=7,
    )
    parent.write_bytes(b"parent-v2")
    changed_parent = mod.radial_task_fingerprint(
        family_id="fixed-family",
        target_radius_mm=81.0,
        parent_paths=(parent,),
        anchor_schedule="balanced",
        robot_config_path=config,
        strategy_version=7,
    )
    config.write_text("robot: v2\n", encoding="utf-8")
    changed_config = mod.radial_task_fingerprint(
        family_id="fixed-family",
        target_radius_mm=81.0,
        parent_paths=(parent,),
        anchor_schedule="balanced",
        robot_config_path=config,
        strategy_version=7,
    )
    changed_schedule = mod.radial_task_fingerprint(
        family_id="fixed-family",
        target_radius_mm=81.0,
        parent_paths=(parent,),
        anchor_schedule="loose",
        robot_config_path=config,
        strategy_version=7,
    )
    changed_radius = mod.radial_task_fingerprint(
        family_id="fixed-family",
        target_radius_mm=81.25,
        parent_paths=(parent,),
        anchor_schedule="balanced",
        robot_config_path=config,
        strategy_version=7,
    )
    changed_version = mod.radial_task_fingerprint(
        family_id="fixed-family",
        target_radius_mm=81.0,
        parent_paths=(parent,),
        anchor_schedule="balanced",
        robot_config_path=config,
        strategy_version=8,
    )

    assert len({base, changed_family, changed_parent, changed_config, changed_schedule, changed_radius, changed_version}) == 7


def test_legacy_artifact_manifest_detects_any_v2_to_v5_mutation(tmp_path: Path) -> None:
    mod = _load_module()
    artifacts = [tmp_path / f"v{version}.json" for version in range(2, 6)]
    for version, path in zip(range(2, 6), artifacts):
        path.write_text(f"v{version}\n", encoding="utf-8")
    before = mod.artifact_hash_manifest(artifacts)

    assert mod.artifact_manifests_match(before, mod.artifact_hash_manifest(artifacts)) is True
    artifacts[-1].write_text("mutated\n", encoding="utf-8")
    assert mod.artifact_manifests_match(before, mod.artifact_hash_manifest(artifacts)) is False


def test_joint_corrector_receives_the_rotated_full_predictor_and_positive_anchor_schedule(monkeypatch) -> None:
    mod = _load_module()
    targets = _path(mod, radius_mm=81.0)[["angle_idx", "angle_rad", "family_id", "radius_mm"]].copy()
    targets["x_target_m"] = targets["angle_idx"] / 1000.0
    targets["y_target_m"] = 0.0
    targets["z_target_m"] = 0.0
    predictor = _path(mod, radius_mm=81.0)
    predictor["radial_predictor_type"] = "parent_copy"
    captured: dict[str, object] = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        frame = kwargs["targets"].copy()
        beta = np.asarray(kwargs["initial_beta"], dtype=float)
        for idx, column in enumerate(mod.atlas.BETA_COLS):
            frame[column] = beta[:, idx]
        return frame, {"centerline_gate_pass": True, "selected_stage": "balanced_1"}

    monkeypatch.setattr(mod.atlas, "optimize_cyclic_trajectory", fake_optimize)

    corrected, report = mod.correct_radial_predictor(
        targets,
        predictor,
        cut_idx=2,
        anchor_schedule="balanced",
        bounds=np.deg2rad(np.asarray([[-30.0, 30.0]] * 6)),
        lengths_m=np.ones(31),
        p_end_local_m=np.asarray([0.0, 0.0, 0.0, 1.0]),
        theta_sign=-1.0,
        max_nfev=5,
        lambda_margin=0.01,
        soft_margin_deg=0.25,
    )

    passed_targets = captured["targets"]
    assert isinstance(passed_targets, pd.DataFrame)
    assert passed_targets["angle_idx"].tolist() == [2, 3, 0, 1]
    expected_beta = mod.rotate_for_cut(predictor, cut_idx=2)[mod.atlas.BETA_COLS].to_numpy(dtype=float)
    assert np.allclose(captured["initial_beta"], expected_beta)
    assert all(float(stage["lambda_anchor"]) > 0.0 for stage in captured["stages"])
    assert all(float(stage["lambda_margin"]) == 0.01 for stage in captured["stages"])
    assert all(float(stage["soft_margin_deg"]) == 0.25 for stage in captured["stages"])
    assert report["lambda_margin"] == 0.01
    assert corrected["angle_idx"].tolist() == [0, 1, 2, 3]
    assert report["full_predictor_path_used"] is True
    assert report["radial_anchor_retained"] is True


def test_candidate_layer_filter_enforces_two_mm_quarter_degree_and_eight_candidate_caps() -> None:
    mod = _load_module()
    rows = []
    for angle_idx in range(2):
        for candidate_idx in range(12):
            row = {
                "angle_idx": angle_idx,
                "xyz_residual_mm": 0.1 + 0.2 * candidate_idx,
            }
            beta = np.deg2rad(np.full(6, candidate_idx * 0.1 + angle_idx))
            for idx, column in enumerate(mod.atlas.BETA_COLS):
                row[column] = beta[idx]
            rows.append(row)
    filtered, report = mod.filter_candidate_layers(
        pd.DataFrame(rows),
        expected_angle_indices=(0, 1),
        residual_limit_mm=2.0,
        cluster_threshold_deg=0.25,
        max_candidates_per_angle=8,
    )

    assert report["candidate_layer_gate_pass"] is True
    assert filtered["xyz_residual_mm"].le(2.0).all()
    assert filtered.groupby("angle_idx").size().le(8).all()
    assert filtered.groupby("angle_idx").size().ge(1).all()


def test_exact_repeatability_is_not_satisfied_by_a_numerically_close_rerun() -> None:
    mod = _load_module()
    first = _path(mod, radius_mm=81.0)
    exact = first.copy()
    close = first.copy()
    close.loc[0, mod.atlas.BETA_COLS[0]] += 1.0e-14

    assert mod.exact_repeatability_report(first, exact)["deterministic_exact_gate_pass"] is True
    assert mod.exact_repeatability_report(first, close)["deterministic_exact_gate_pass"] is False


def test_repeat_input_tracks_candidate_graph_rescue_instead_of_original_predictor() -> None:
    mod = _load_module()
    direct = pd.DataFrame({"angle_idx": [0], "beta1_rad": [1.0]})
    rescue = pd.DataFrame({"angle_idx": [0], "beta1_rad": [2.0]})

    selected = mod.repeat_input_for_strategy(
        direct,
        selected_strategy="candidate_graph_joint_corrector",
        candidate_graph_predictor=rescue,
    )

    pd.testing.assert_frame_equal(selected, rescue)
    assert selected is not rescue
