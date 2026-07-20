from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.student import (
    assign_dense_phase_split,
    materialize_phase_splits,
    periodic_beta_interpolation,
)


try:
    import tensorflow as tf
except Exception:
    tf = None

TF_READY = tf is not None and all(
    hasattr(tf, name) for name in ("constant", "zeros", "keras")
)
requires_tensorflow = pytest.mark.skipif(
    not TF_READY,
    reason="TensorFlow is unavailable or its namespace is incomplete",
)

from quasi_exp.teacher.student_tracking_tf import (
    StudentGeometry,
    autoregressive_rollout,
    build_gru_model,
    build_static_model,
    compile_student,
    decode_beta_tf,
    make_cyclic_windows,
    student_loss_terms,
)
from scripts.analysis.verify_large_scale_student_artifacts_v10 import verify


def test_periodic_initial_path_interpolates_across_the_cyclic_seam() -> None:
    phase = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    frame = pd.DataFrame({"phase_rad": phase})
    for index in range(1, 7):
        frame[f"teacher_beta{index}_rad"] = np.cos(phase) * index

    dense = periodic_beta_interpolation(frame, phase_count=8)

    assert dense.shape == (8, 6)
    np.testing.assert_allclose(dense[0], np.arange(1.0, 7.0))
    # Linear periodic interpolation at 7π/4 lies halfway between 0 and 3π/2.
    np.testing.assert_allclose(dense[-1], 0.5 * np.arange(1.0, 7.0))


def test_dense_phase_split_is_disjoint_and_excludes_failed_teacher_labels() -> None:
    rows = 720
    frame = pd.DataFrame(
        {
            "phase_idx": np.arange(rows),
            "teacher_fk_residual_mm": np.full(rows, 0.2),
            "within_joint_bounds": np.ones(rows, dtype=bool),
        }
    )
    frame.loc[13, "teacher_fk_residual_mm"] = 16.0
    frame.loc[28, "within_joint_bounds"] = False

    split = assign_dense_phase_split(frame, expected_phase_count=720)

    assert split["split"].value_counts().to_dict() == {
        "train": 360,
        "validation": 180,
        "test": 180,
    }
    assert split.loc[split.phase_idx == 13, "split"].item() == "validation"
    assert split.loc[split.phase_idx == 13, "label_eligible"].item() is False
    assert split.loc[split.phase_idx == 28, "label_eligible"].item() is False
    assert not split.loc[split.split.eq("test"), "used_for_training"].any()
    assert split.loc[split.used_for_training, "split"].eq("train").all()
    assert split.loc[split.used_for_training, "label_eligible"].all()


def test_materialized_test_labels_are_physically_separate_from_training(
    tmp_path,
) -> None:
    rows = 720
    frame = pd.DataFrame(
        {
            "phase_idx": np.arange(rows),
            "teacher_fk_residual_mm": np.full(rows, 0.1),
            "within_joint_bounds": np.ones(rows, dtype=bool),
            "teacher_beta1_rad": np.arange(rows, dtype=float),
        }
    )
    split = assign_dense_phase_split(frame)

    report = materialize_phase_splits(split, output_dir=tmp_path)

    train = pd.read_parquet(tmp_path / "train.parquet")
    validation = pd.read_parquet(tmp_path / "validation.parquet")
    test = pd.read_parquet(tmp_path / "sealed_test.parquet")
    assert len(train) == 360
    assert len(validation) == 180
    assert len(test) == 180
    assert set(train.phase_idx).isdisjoint(test.phase_idx)
    assert set(validation.phase_idx).isdisjoint(test.phase_idx)
    assert report["sealed_test_label_access"] == "post_selection_only"
    assert report["split_sha256"]["test"]


@requires_tensorflow
def test_tanh_beta_decoder_stays_strictly_inside_registered_bounds() -> None:
    bounds = np.column_stack([np.full(6, -0.2), np.full(6, 0.3)]).astype(np.float32)
    decoded = decode_beta_tf(
        tf.constant([[-100.0] * 6, [100.0] * 6]), bounds, "tanh"
    ).numpy()

    assert np.all(decoded >= bounds[:, 0])
    assert np.all(decoded <= bounds[:, 1])


@requires_tensorflow
def test_fk_aware_loss_is_zero_for_exact_beta_and_target() -> None:
    lengths = np.full(31, 0.01, dtype=np.float32)
    endpoint = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    geometry = StudentGeometry(
        lengths_m=lengths,
        p_end_local_m=endpoint,
        theta_sign=-1.0,
        beta_bounds_rad=np.column_stack([np.full(6, -0.5), np.full(6, 0.5)]),
    )
    beta = tf.zeros((2, 6), dtype=tf.float32)
    xyz = tf.constant([[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=tf.float32)
    packed = tf.concat([beta, xyz], axis=1)

    terms = student_loss_terms(packed, beta, geometry=geometry, lambda_fk=1.0)

    assert terms["total"].numpy() == pytest.approx(0.0, abs=1.0e-9)
    assert terms["fk"].numpy() == pytest.approx(0.0, abs=1.0e-9)


def test_gru_rollout_feeds_back_predictions_without_teacher_injection() -> None:
    class IncrementPreviousBeta:
        def __call__(self, features, training=False):
            del training
            array = np.asarray(features, dtype=np.float32)
            output = array[..., 6:12].copy()
            output[:, -1, :] += 0.1
            return output

    target = np.column_stack([np.arange(4, dtype=float), np.zeros((4, 2))])
    predicted = autoregressive_rollout(
        IncrementPreviousBeta(),
        target,
        initial_beta_rad=np.zeros(6),
        window_size=2,
    )

    np.testing.assert_allclose(predicted[:, 0], [0.1, 0.2, 0.3, 0.4], atol=1.0e-7)
    np.testing.assert_allclose(predicted, np.repeat(predicted[:, :1], 6, axis=1), atol=1.0e-7)


@requires_tensorflow
def test_static_student_accepts_packed_beta_and_xyz_targets(tmp_path) -> None:
    geometry = StudentGeometry(
        lengths_m=np.full(31, 0.01, dtype=np.float32),
        p_end_local_m=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        theta_sign=-1.0,
        beta_bounds_rad=np.column_stack([np.full(6, -0.5), np.full(6, 0.5)]),
    )
    xyz = np.array([[0.3, 0.0, 0.0], [0.3, 0.001, 0.0]], dtype=np.float32)
    packed = np.column_stack([np.zeros((2, 6), dtype=np.float32), xyz])
    model = build_static_model(xyz, geometry=geometry, output_mode="tanh", hidden_units=(8,))
    compile_student(model, geometry=geometry, lambda_fk=0.1, learning_rate=1.0e-3)

    loss = model.train_on_batch(xyz, packed)

    assert np.isfinite(loss)
    assert model(xyz).shape == (2, 6)
    model.save(tmp_path / "student.keras")
    restored = tf.keras.models.load_model(tmp_path / "student.keras", compile=False)
    assert restored(xyz).shape == (2, 6)


def test_gru_windows_cover_cyclic_cuts_in_both_directions() -> None:
    rows = 8
    frame = pd.DataFrame(
        {
            "phase_idx": np.arange(rows),
            "target_x_m": np.arange(rows, dtype=float),
            "target_y_m": np.zeros(rows),
            "target_z_m": np.zeros(rows),
        }
    )
    for index in range(1, 7):
        frame[f"teacher_beta{index}_rad"] = np.arange(rows, dtype=float) + index

    features, targets = make_cyclic_windows(frame, window_size=4, include_reverse=True)

    assert features.shape == (16, 4, 12)
    assert targets.shape == (16, 4, 9)
    # Forward cut 0 sees phase 7 as its cyclic predecessor.
    np.testing.assert_allclose(features[0, 0, 6:12], np.arange(8.0, 14.0))
    # The first reverse window traverses 7,6,5,4.
    np.testing.assert_allclose(targets[8, :, 6], [7.0, 6.0, 5.0, 4.0])


@requires_tensorflow
def test_gru_student_trains_on_sequence_packed_targets() -> None:
    geometry = StudentGeometry(
        lengths_m=np.full(31, 0.01, dtype=np.float32),
        p_end_local_m=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        theta_sign=-1.0,
        beta_bounds_rad=np.column_stack([np.full(6, -0.5), np.full(6, 0.5)]),
    )
    features = np.zeros((3, 4, 12), dtype=np.float32)
    targets = np.zeros((3, 4, 9), dtype=np.float32)
    targets[..., 6] = 0.3
    model = build_gru_model(features, geometry=geometry, units=4)
    compile_student(model, geometry=geometry, lambda_fk=0.1, learning_rate=1.0e-3)

    loss = model.train_on_batch(features, targets)

    assert np.isfinite(loss)
    assert model(features).shape == (3, 4, 6)


def test_artifact_verifier_fails_closed_on_hash_mismatch(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"registered evidence")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert verify(artifact, digest)["sha256"] == digest
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify(artifact, "0" * 64)
