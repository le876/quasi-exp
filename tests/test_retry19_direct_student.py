from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS
from quasi_exp.teacher.retry19_direct_student import (
    DirectStudentConfig,
    aligned_q0_augmented_and_f0_rows,
    build_direct_student,
    classify_causal_evidence,
    direct_feature_matrix,
    edge_delta_loss_numpy,
    load_direct_student,
    smooth_signed_power,
    smooth_signed_power_derivative,
    train_direct_student,
    unified_split_registry,
)


def test_smooth_signed_power_is_odd_and_has_finite_zero_gradient() -> None:
    values = np.asarray([-1.0e-8, 0.0, 1.0e-8])
    transformed = smooth_signed_power(values, alpha=0.5, epsilon=0.015)
    derivative = smooth_signed_power_derivative(values, alpha=0.5, epsilon=0.015)
    assert transformed[0] == pytest.approx(-transformed[2])
    assert transformed[1] == 0.0
    assert np.isfinite(derivative).all()
    assert derivative[1] == pytest.approx(1 / np.sqrt(0.015))


def test_direct_features_keep_raw_signed_coordinates() -> None:
    xyz = np.asarray([[1.0, -0.002, 0.003], [1.0, 0.002, -0.003]])
    feature = direct_feature_matrix(
        xyz,
        zero_x_m=1.2,
        axial_scale_mm=600,
        radial_scale_mm=200,
        signed_power_alpha=0.5,
        signed_power_epsilon_mm=3.0,
    )
    assert feature.shape == (2, 5)
    assert feature[0, 1] < 0 < feature[1, 1]
    assert feature[0, 2] > 0 > feature[1, 2]


def test_q0_augmented_and_f0_are_row_weight_and_batch_aligned() -> None:
    frame = pd.DataFrame(
        [
            {
                "x_m": 1.0,
                "y_m": -0.1,
                "z_m": 0.2,
                **dict.fromkeys(BETA_COLUMNS, 0.1),
                "orbit_size": 4,
                "sample_weight": 2.0,
            },
            {
                "x_m": 0.9,
                "y_m": 0.1,
                "z_m": -0.2,
                **dict.fromkeys(BETA_COLUMNS, 0.2),
                "orbit_size": 4,
                "sample_weight": 2.0,
            },
        ]
    )
    q0, f0 = aligned_q0_augmented_and_f0_rows(frame)
    assert q0["control_row_id"].tolist() == f0["control_row_id"].tolist()
    assert q0["control_row_ordinal"].tolist() == f0["control_row_ordinal"].tolist()
    assert q0["orbit_normalized_weight"].tolist() == f0["orbit_normalized_weight"].tolist()
    assert (q0[["y_m", "z_m"]] >= 0).all().all()
    assert (f0["y_m"] < 0).any() and (f0["z_m"] < 0).any()


def test_unified_split_inherits_signed_macroblock_and_excludes_audit_rows() -> None:
    history = pd.DataFrame(
        [
            {"x_m": 0.01, "y_m": -0.01, "z_m": 0.01, "split_role": "validation"},
        ]
    )
    current = pd.DataFrame(
        [
            {"target_id": "same", "x_m": 0.02, "y_m": -0.02, "z_m": 0.02, "domain_class": "primary"},
            {"target_id": "audit", "x_m": 0.02, "y_m": -0.02, "z_m": 0.02, "domain_class": "audit_only"},
        ]
    )
    assigned, registry = unified_split_registry(current, history)
    assert assigned.set_index("target_id").loc["same", "split_role"] == "validation"
    assert assigned.set_index("target_id").loc["audit", "split_role"] == "audit_only"
    assert registry["assignment_reason"].eq("inherited_retry18_macroblock").all()


def test_unified_split_reassigns_conflicting_historical_macroblock_atomically() -> None:
    history = pd.DataFrame(
        [
            {"x_m": 0.01, "y_m": -0.01, "z_m": 0.01, "split_role": "train"},
            {"x_m": 0.02, "y_m": -0.02, "z_m": 0.02, "split_role": "test"},
        ]
    )
    current = pd.DataFrame(
        [
            {"target_id": "a", "x_m": 0.01, "y_m": -0.01, "z_m": 0.01, "domain_class": "primary"},
            {"target_id": "b", "x_m": 0.02, "y_m": -0.02, "z_m": 0.02, "domain_class": "primary"},
        ]
    )

    assigned, registry = unified_split_registry(current, history, seed=20260950)
    repeated, repeated_registry = unified_split_registry(current, history, seed=20260950)

    assert assigned["split_role"].nunique() == 1
    assert registry["assignment_reason"].tolist() == [
        "retry18_conflict_reassigned_by_retry19_seed_hash"
    ]
    pd.testing.assert_frame_equal(assigned, repeated)
    pd.testing.assert_frame_equal(registry, repeated_registry)


def test_unified_split_does_not_promote_posthoc_orbit_audit_to_a_hard_gate() -> None:
    history = pd.DataFrame(
        [
            {"x_m": 0.01, "y_m": 0.0, "z_m": 0.0, "split_role": "train", "symmetry_orbit_id": "orbit"},
            {"x_m": 0.05, "y_m": 0.0, "z_m": 0.0, "split_role": "test", "symmetry_orbit_id": "orbit"},
        ]
    )
    current = history.assign(target_id=["a", "b"], domain_class="primary").drop(columns="split_role")

    assigned, _registry = unified_split_registry(current, history)

    assert assigned.groupby("macroblock_id")["split_role"].nunique().max() == 1
    assert assigned.groupby("symmetry_orbit_id")["split_role"].nunique().item() == 2


def test_edge_loss_matches_teacher_delta_not_output_smoothing() -> None:
    xyz = np.asarray([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
    teacher = np.zeros((2, 6))
    teacher[1, 0] = 0.2
    prediction = teacher + 1.0  # large absolute offset, exact local delta
    loss = edge_delta_loss_numpy(prediction, teacher, xyz, np.asarray([[0, 1]]))
    assert loss == pytest.approx(0.0)
    smoothed = np.ones((2, 6))
    assert edge_delta_loss_numpy(smoothed, teacher, xyz, np.asarray([[0, 1]])) > 0


def test_causal_output_allows_mixed_factors() -> None:
    result = classify_causal_evidence(
        {
            "q0_augmented_raw_max_spike_mm": 10.0,
            "f0_direct_raw_max_spike_mm": 4.0,
            "f0_nonregression": True,
            "best_without_jacobian_raw_max_spike_mm": 4.0,
            "jacobian_raw_max_spike_mm": 2.0,
            "jacobian_crosses_raw_gate": True,
            "replicated_seed_pass_count": 3,
        }
    )
    assert result["primary_cause_class"] == "mixed_representation_and_jacobian"
    assert result["causal_evidence_strength"] == "strong"


def test_direct_keras_feature_gradient_is_finite_at_zero_and_model_is_bounded(tmp_path) -> None:
    import tensorflow as tf

    config = DirectStudentConfig(
        hidden_units=(8,),
        maximum_steps=2,
        validation_interval=1,
        patience_intervals=1,
        batch_size=4,
        radial_scale_mm=200.0,
        signed_power_alpha=0.5,
        signed_power_epsilon_mm=3.0,
    )
    bounds = np.tile(np.asarray([[-0.5, 0.5]]), (6, 1))
    train_xyz = np.asarray([[1.2, 0.0, 0.0], [1.1, 0.01, -0.01]], dtype=np.float32)
    model = build_direct_student(
        train_xyz_m=train_xyz,
        zero_x_m=1.2,
        beta_bounds_rad=bounds,
        config=config,
        include_signed_power=True,
    )
    point = tf.Variable([[1.2, 0.0, 0.0]], dtype=tf.float32)
    with tf.GradientTape() as tape:
        value = tf.reduce_sum(model(point, training=False))
    gradient = tape.gradient(value, point).numpy()
    prediction = model(train_xyz, training=False).numpy()
    assert np.isfinite(gradient).all()
    assert np.isfinite(prediction).all()
    assert np.max(prediction) <= 0.5 + 1.0e-6
    assert np.min(prediction) >= -0.5 - 1.0e-6
    path = tmp_path / "direct.keras"
    model.save(path)
    restored = load_direct_student(path)
    assert np.allclose(restored(train_xyz, training=False), prediction, atol=1.0e-6, rtol=0)


def test_direct_trainer_defaults_missing_row_weights_to_one() -> None:
    config = DirectStudentConfig(
        hidden_units=(8,),
        maximum_steps=2,
        validation_interval=1,
        patience_intervals=2,
        batch_size=4,
    )
    rows = []
    for index in range(6):
        rows.append(
            {
                "target_id": f"target-{index}",
                "x_m": 1.2 - 0.01 * index,
                "y_m": 0.001 * index,
                "z_m": -0.001 * index,
                "sample_weight": np.nan if index % 2 == 0 else 1.0,
                **{column: 0.01 * (index + beta_index) for beta_index, column in enumerate(BETA_COLUMNS)},
            }
        )
    frame = pd.DataFrame(rows)

    model, history = train_direct_student(
        frame.iloc[:4],
        frame.iloc[4:],
        pd.DataFrame(),
        zero_x_m=1.2,
        beta_bounds_rad=np.tile(np.asarray([[-0.5, 0.5]]), (6, 1)),
        config=config,
        include_signed_power=False,
    )

    prediction = np.asarray(model(frame.loc[:, ["x_m", "y_m", "z_m"]].to_numpy(np.float32), training=False))
    assert np.isfinite(prediction).all()
    assert np.isfinite(history.select_dtypes(include=[np.number]).to_numpy(float)).all()
