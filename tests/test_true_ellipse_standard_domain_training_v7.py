from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


class _IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)

    def inverse_transform(self, values):
        return np.asarray(values, dtype=float)


class _ZeroModel:
    n_iter_ = 1
    loss_ = 0.0

    def predict(self, features):
        return np.zeros((len(features), 6), dtype=float)


def _load_module():
    analysis_dir = REPO_ROOT / "scripts" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    path = analysis_dir / "run_true_ellipse_standard_domain_training_v7.py"
    spec = importlib.util.spec_from_file_location(
        "run_true_ellipse_standard_domain_training_v7", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registered_screen_crosses_24_base_configs_with_two_output_links() -> None:
    mod = _load_module()

    configs = mod.model_configs()
    ids = [config.config_id for config in configs]

    assert len(configs) == 48
    assert len(set(ids)) == 48
    assert {config.output_link_id for config in configs} == {"identity", "tanh_bounds"}
    assert all(sum(config.base_config == other.base_config for other in configs) == 2 for config in configs)
    assert any(
        config.base_config.config_id == mod.V4_BASELINE_CONFIG_ID
        and config.output_link_id == "identity"
        for config in configs
    )
    assert any(
        config.base_config.config_id == mod.V4_BASELINE_CONFIG_ID
        and config.output_link_id == "tanh_bounds"
        for config in configs
    )


def test_output_link_training_targets_are_encoded_without_clipping() -> None:
    mod = _load_module()
    domain = mod.engine.registered_joint_domain("standard_beta34_10deg_v1")
    beta = np.deg2rad(
        np.asarray(
            [
                [0.0, 1.0, -2.0, 3.0, -4.0, 5.0],
                [4.0, -4.0, 9.0, -9.0, 14.0, -14.0],
            ]
        )
    )

    latent, report = mod.encode_training_targets(
        beta,
        output_link_id="tanh_bounds",
        domain=domain,
    )
    decoded = mod.engine.registered_output_link("tanh_bounds").decode(latent, domain=domain)

    assert report["target_link_clip_count"] == 0
    assert np.allclose(decoded, beta)


def test_deployed_centerline_gate_requires_raw_bounds_and_positive_margin() -> None:
    mod = _load_module()
    passing = {
        "beta_p95_deg": 0.1,
        "ee_p95_mm": 1.0,
        "ee_max_mm": 2.0,
        "axiserr_max_p95_abs_mm": 1.0,
        "fixed_any_axis_bias_gt2mm": False,
        "delta_beta_p95_deg": 0.1,
        "delta_beta_max_deg": 0.2,
        "delta2_beta_p95_deg": 0.05,
        "seam_beta_rms_deg": 0.1,
        "beta_bound_violation_count": 0,
        "prediction_min_joint_margin_deg": 0.01,
    }

    assert mod.deployed_centerline_model_gate(passing) is True
    outside = dict(passing, beta_bound_violation_count=1)
    shallow = dict(passing, prediction_min_joint_margin_deg=0.0099)
    assert mod.deployed_centerline_model_gate(outside) is False
    assert mod.deployed_centerline_model_gate(shallow) is False


def test_nonformal_evaluation_never_registers_test_or_test_challenge() -> None:
    mod = _load_module()
    holdout = {"validation_radius_mm": 102.5, "test_radius_mm": 110.0}

    smoke = mod.final_evaluation_specs(preset="smoke", holdout=holdout)
    formal = mod.final_evaluation_specs(preset="formal", holdout=holdout)

    assert {spec["split"] for spec in smoke} == {"validation"}
    assert all("test" not in spec["label"] for spec in smoke)
    assert {spec["split"] for spec in formal} == {"validation", "test"}
    assert "test_half_phase" in {spec["label"] for spec in formal}
    assert "test_integer_centerline" in {spec["label"] for spec in formal}
    assert "test_tube_diagnostic" in {spec["label"] for spec in formal}


def test_nonformal_training_sources_use_safe_dataset_and_validation_challenge_only(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    full = tmp_path / "full.parquet"
    safe = tmp_path / "safe.parquet"
    validation = tmp_path / "validation.parquet"
    test = tmp_path / "test.parquet"
    for path in (full, safe, validation, test):
        path.write_bytes(path.name.encode())
    upstream = {
        "dataset_path": str(full),
        "dataset_sha256": mod.file_sha256(full),
        "nonformal_dataset_path": str(safe),
        "nonformal_dataset_sha256": mod.file_sha256(safe),
        "challenge_paths": {"validation": str(validation), "test": str(test)},
        "challenge_reports": {
            "validation": {"challenge_artifact_sha256": mod.file_sha256(validation)},
            "test": {"challenge_artifact_sha256": mod.file_sha256(test)},
        },
    }

    smoke = mod.resolve_training_sources(
        Namespace(preset="smoke", tube_dataset=full), upstream
    )
    formal = mod.resolve_training_sources(
        Namespace(preset="formal", tube_dataset=full), upstream
    )

    assert smoke["dataset_path"] == safe
    assert smoke["dataset_sha256"] == mod.file_sha256(safe)
    assert set(smoke["challenge_paths"]) == {"validation"}
    assert test not in smoke["challenge_paths"].values()
    assert formal["dataset_path"] == full
    assert set(formal["challenge_paths"]) == {"validation", "test"}


def test_formal_model_gate_requires_stable_integer_and_half_phase_but_not_tube() -> None:
    mod = _load_module()
    stable = {"stable_gate_pass": True, "passed_seed_count": 4, "total_seed_count": 5}
    failed = {"stable_gate_pass": False, "passed_seed_count": 3, "total_seed_count": 5}
    gates = {
        "validation_integer_centerline": stable,
        "validation_half_phase": stable,
        "test_integer_centerline": stable,
        "test_half_phase": stable,
        "validation_tube_diagnostic": failed,
        "test_tube_diagnostic": failed,
    }

    assert mod.formal_model_gate_pass(
        formal_claims_allowed=True,
        seed_protocol_pass=True,
        target_link_clip_count=0,
        gates=gates,
    ) is True
    broken = dict(gates, test_half_phase=failed)
    assert mod.formal_model_gate_pass(
        formal_claims_allowed=True,
        seed_protocol_pass=True,
        target_link_clip_count=0,
        gates=broken,
    ) is False
    assert mod.formal_model_gate_pass(
        formal_claims_allowed=True,
        seed_protocol_pass=True,
        target_link_clip_count=1,
        gates=gates,
    ) is False


def test_training_assignment_preserves_upstream_dynamic_split_and_no_centerline_training() -> None:
    mod = _load_module()
    dataset = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "family_id": ["f"] * 4,
            "trajectory_id": ["f@75", "f@75", "f@102.5", "f@110"],
            "radius_mm": [75.0, 75.0, 102.5, 110.0],
            "angle_idx": [0, 1, 0, 0],
            "is_centerline": [False, True, True, True],
            "split": ["train", "train", "validation", "test"],
            "used_for_training": [True, False, False, False],
        }
    )

    assignment, report = mod.make_training_assignment(
        dataset,
        validation_radius_mm=102.5,
        test_radius_mm=110.0,
    )

    assert report["split_gate_pass"] is True
    assert assignment.loc[assignment["used_for_training"], "sample_id"].tolist() == ["a"]
    assert not assignment.loc[assignment["used_for_training"], "is_centerline"].any()


def test_nonformal_training_assignment_accepts_physically_absent_test_rows() -> None:
    mod = _load_module()
    dataset = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "family_id": ["f"] * 3,
            "trajectory_id": ["f@75", "f@75", "f@102.5"],
            "radius_mm": [75.0, 75.0, 102.5],
            "angle_idx": [0, 1, 0],
            "is_centerline": [False, True, True],
            "split": ["train", "train", "validation"],
            "used_for_training": [True, False, False],
        }
    )

    _assignment, report = mod.make_training_assignment(
        dataset,
        validation_radius_mm=102.5,
        test_radius_mm=110.0,
        require_test=False,
    )

    assert report["validation_present"] is True
    assert report["test_present"] is False
    assert report["split_gate_pass"] is True


def test_nonformal_final_fingerprint_never_opens_full_registered_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    out_dir = tmp_path / "training"
    (out_dir / "00_audit").mkdir(parents=True)
    (out_dir / "01_split").mkdir(parents=True)
    mod.write_json(
        out_dir / "00_audit" / "audit_report.json",
        {
            "holdout": {"validation_radius_mm": 102.5, "test_radius_mm": 110.0},
            "dataset_sha256": "safe-dataset-hash",
        },
    )
    assignment = out_dir / "01_split" / "split_assignment.parquet"
    assignment.write_bytes(b"assignment")
    full = tmp_path / "full-registered.parquet"
    calls = []
    real_hash = mod.file_sha256

    def recording_hash(path):
        calls.append(Path(path))
        return real_hash(path)

    monkeypatch.setattr(mod, "file_sha256", recording_hash)
    args = Namespace(
        out_dir=out_dir,
        tube_dataset=full,
        preset="smoke",
        seeds=",".join(str(seed) for seed in mod.FORMAL_SEEDS),
    )

    mod._final_task_fingerprint(
        args,
        {"task_fingerprint": "screen", "selected_config": {"id": "config"}},
    )

    assert full not in calls


def test_training_worker_encodes_decodes_and_evaluates_registered_challenge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    beta_columns = mod.BETA_COLUMNS
    rows = []
    for angle in range(8):
        row = {
            "sample_id": f"train:{angle}",
            "family_id": "f",
            "trajectory_id": "f@75",
            "radius_mm": 75.0,
            "angle_idx": angle,
            "angle_rad": angle * 2.0 * np.pi / 8.0,
            "is_centerline": False,
            "split": "train",
            "used_for_training": True,
            "x_target_m": 0.001 * angle,
            "y_target_m": 0.0,
            "z_target_m": 0.0,
        }
        row.update({column: 0.0 for column in beta_columns})
        rows.append(row)
    dataset = pd.DataFrame(rows)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    assignment = dataset[
        [
            "sample_id",
            "family_id",
            "trajectory_id",
            "radius_mm",
            "angle_idx",
            "is_centerline",
            "split",
            "used_for_training",
        ]
    ].copy()
    assignment_path = tmp_path / "assignment.parquet"
    assignment.to_parquet(assignment_path, index=False)
    challenge = pd.DataFrame(
        {
            "family_id": ["f"] * 4,
            "radius_mm": [82.5] * 4,
            "angle_idx": np.arange(4),
            "angle_rad": np.arange(4) * np.pi / 2.0,
            "split": ["validation"] * 4,
            "x_target_m": [0.0] * 4,
            "y_target_m": [0.0] * 4,
            "z_target_m": [0.0] * 4,
            **{column: [0.0] * 4 for column in beta_columns},
        }
    )
    challenge_path = tmp_path / "validation_challenge.parquet"
    challenge.to_parquet(challenge_path, index=False)
    robot_config = REPO_ROOT / "configs" / "robot_rods_only_standard_100k.yaml"
    shared = mod.TrainingInputs(
        dataset_path=dataset_path,
        dataset_sha256=mod.file_sha256(dataset_path),
        assignment_path=assignment_path,
        assignment_sha256=mod.file_sha256(assignment_path),
        robot_config_path=robot_config,
        robot_config_sha256=mod.file_sha256(robot_config),
        theta_sign=-1.0,
        family_id="f",
        challenge_paths={"validation": challenge_path},
        challenge_hashes={"validation": mod.file_sha256(challenge_path)},
    )
    args = Namespace(preset="smoke")
    config = mod.LinkedModelConfig(
        base_config=mod.v5.v5_model_configs()[0],
        output_link_id="tanh_bounds",
    )
    task = mod._training_task(
        args,
        shared=shared,
        task_id="worker_test",
        mode="test",
        config=config,
        seed=1,
        result_path=tmp_path / "result.json",
        package_path=None,
        prediction_dir=None,
        angle_stride=1,
        max_iter=1,
        evaluation_specs=[
            {
                "label": "validation_half_phase",
                "split": "validation",
                "radius_mm": 82.5,
                "source": "challenge",
                "evaluation_kind": "half_phase_centerline",
            }
        ],
    )

    monkeypatch.setattr(
        mod.v5.v4,
        "_fit_scaled_model",
        lambda **_kwargs: (_ZeroModel(), _IdentityScaler(), _IdentityScaler(), [], 0.0),
    )
    result = mod.run_training_worker(task)

    assert result["output_link_id"] == "tanh_bounds"
    assert result["target_link_clip_count"] == 0
    metrics = result["evaluations"]["validation_half_phase"]
    assert metrics["eval_rows"] == 4
    assert metrics["beta_bound_violation_count"] == 0
    assert metrics["prediction_min_joint_margin_deg"] == 5.0
