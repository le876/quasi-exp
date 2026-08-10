"""Gold-set-aware cyclic assignment and Student fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .canonical import link_cyclic_candidates
from .dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from .student_tracking_tf import StudentGeometry


def _rms_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    return np.rad2deg(
        np.sqrt(np.mean(np.square(lhs - rhs), axis=-1))
    )


def assign_cyclic_gold_section(
    prediction_beta_rad: np.ndarray,
    gold_candidates: pd.DataFrame,
    *,
    max_transition_deg: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select the closest complete cyclic section in a finite Gold graph."""

    prediction = np.asarray(prediction_beta_rad, dtype=float)
    required = {
        "phase_idx",
        "candidate_idx",
        "residual_mm",
        "minimum_joint_margin_deg",
        *BETA_COLUMNS,
    }
    missing = sorted(required - set(gold_candidates.columns))
    if missing:
        raise ValueError(f"Gold candidates missing columns: {missing}")
    gold = gold_candidates.sort_values(
        ["phase_idx", "candidate_idx", "residual_mm", "source"],
        kind="stable",
    ).reset_index(drop=True)
    phases = sorted(map(int, gold["phase_idx"].unique()))
    if phases != list(range(len(phases))):
        raise ValueError("Gold candidate phases must be complete and dense")
    if prediction.shape != (len(phases), 6):
        raise ValueError(
            "prediction must have shape (phase_count, 6); "
            f"received {prediction.shape}"
        )

    phase_frames: list[pd.DataFrame] = []
    candidate_layers: list[np.ndarray] = []
    unary_layers: list[np.ndarray] = []
    for phase_idx in phases:
        frame = gold.loc[gold["phase_idx"].eq(phase_idx)].reset_index(
            drop=True
        )
        beta = frame.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
        phase_frames.append(frame)
        candidate_layers.append(beta)
        unary_layers.append(_rms_deg(beta, prediction[phase_idx][None, :]))

    selected_beta, link_report = link_cyclic_candidates(
        candidate_layers,
        unary_layers,
        lambda_velocity=1.0e-9,
        closure_weight=0.0,
        lambda_posture=0.0,
        max_transition_deg=float(max_transition_deg),
    )
    if not bool(link_report.get("success", False)):
        return pd.DataFrame(), {
            **link_report,
            "phase_count": len(phases),
            "max_transition_deg": float(max_transition_deg),
        }

    selected_rows: list[dict[str, Any]] = []
    assignment_distance: list[float] = []
    for phase_idx, (frame, layer) in enumerate(
        zip(phase_frames, candidate_layers)
    ):
        gap = _rms_deg(layer, selected_beta[phase_idx][None, :])
        position = min(
            range(len(frame)),
            key=lambda index: (
                float(gap[index]),
                float(frame.iloc[index]["residual_mm"]),
                int(frame.iloc[index]["candidate_idx"]),
            ),
        )
        row = frame.iloc[int(position)].to_dict()
        row["assignment_rms_deg"] = float(
            _rms_deg(
                selected_beta[phase_idx],
                prediction[phase_idx],
            )
        )
        selected_rows.append(row)
        assignment_distance.append(float(row["assignment_rms_deg"]))

    selected = pd.DataFrame(selected_rows).sort_values(
        "phase_idx", kind="stable"
    )
    transition = _rms_deg(
        np.roll(selected_beta, -1, axis=0), selected_beta
    )
    return selected, {
        **link_report,
        "phase_count": len(phases),
        "max_transition_deg": float(max_transition_deg),
        "assignment_rms_p50_deg": float(
            np.percentile(assignment_distance, 50)
        ),
        "assignment_rms_p95_deg": float(
            np.percentile(assignment_distance, 95)
        ),
        "assignment_rms_max_deg": float(np.max(assignment_distance)),
        "selected_residual_p95_mm": float(
            np.percentile(selected["residual_mm"], 95)
        ),
        "selected_residual_max_mm": float(selected["residual_mm"].max()),
        "selected_minimum_joint_margin_deg": float(
            selected["minimum_joint_margin_deg"].min()
        ),
        "selected_transition_p95_deg": float(
            np.percentile(transition, 95)
        ),
        "selected_transition_max_deg": float(np.max(transition)),
        "selected_seam_deg": float(transition[-1]),
    }


def _huber(values: Any) -> Any:
    import tensorflow as tf

    absolute = tf.abs(values)
    return tf.where(
        absolute <= 1.0,
        0.5 * tf.square(values),
        absolute - 0.5,
    )


def fine_tune_gold_set_student(
    model: Any,
    training_frame: pd.DataFrame,
    *,
    geometry: StudentGeometry,
    seed: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    beta_loss_scale_deg: Sequence[float],
    lambda_fk: float,
    lambda_margin: float,
    lambda_continuity: float,
    margin_target_deg: float,
    transition_target_deg: float,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Fine-tune one model against one frozen latent Gold section."""

    import tensorflow as tf
    from quasi_exp.model.kinematics_tf import forward_xyz_from_beta_tf

    required = {
        "family_id",
        "phase_idx",
        *XYZ_COLUMNS,
        *BETA_COLUMNS,
    }
    missing = sorted(required - set(training_frame.columns))
    if missing:
        raise ValueError(f"training frame missing columns: {missing}")
    frame = training_frame.sort_values(
        ["family_id", "phase_idx"], kind="stable"
    ).reset_index(drop=True)
    family_sizes = frame.groupby("family_id", sort=True).size()
    if family_sizes.nunique() != 1:
        raise ValueError("all cyclic training families must have equal length")
    family_count = int(len(family_sizes))
    phase_count = int(family_sizes.iloc[0])
    expected_phase = np.tile(np.arange(phase_count), family_count)
    if not np.array_equal(
        frame["phase_idx"].to_numpy(dtype=int), expected_phase
    ):
        raise ValueError("each training family must contain dense phases")

    tf.keras.utils.set_random_seed(int(seed))
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    beta = frame.loc[:, BETA_COLUMNS].to_numpy(dtype=np.float32)
    scale = tf.constant(
        np.deg2rad(np.asarray(beta_loss_scale_deg, dtype=np.float32)),
        dtype=tf.float32,
    )
    bounds = tf.constant(
        np.asarray(geometry.beta_bounds_rad, dtype=np.float32),
        dtype=tf.float32,
    )
    margin_target = tf.constant(
        np.deg2rad(float(margin_target_deg)), dtype=tf.float32
    )
    transition_target = tf.constant(
        float(transition_target_deg), dtype=tf.float32
    )
    optimizer = tf.keras.optimizers.Adam(float(learning_rate))
    dataset = (
        tf.data.Dataset.from_tensor_slices((xyz, beta))
        .shuffle(len(frame), seed=int(seed), reshuffle_each_iteration=True)
        .batch(int(batch_size))
    )

    def point_loss(batch_xyz: Any, batch_beta: Any, training: bool) -> tuple[Any, ...]:
        prediction = model(batch_xyz, training=training)
        beta_loss = tf.reduce_mean(
            _huber((prediction - batch_beta) / scale)
        )
        predicted_xyz = forward_xyz_from_beta_tf(
            prediction,
            lengths_m=geometry.lengths_m,
            p_end_local_m=geometry.p_end_local_m,
            theta_sign=geometry.theta_sign,
        )
        fk_scaled = (
            predicted_xyz - tf.cast(batch_xyz, prediction.dtype)
        ) * tf.cast(1000.0 / 3.0, prediction.dtype)
        fk_loss = tf.reduce_mean(_huber(fk_scaled))
        margin = tf.minimum(
            prediction - bounds[:, 0],
            bounds[:, 1] - prediction,
        )
        margin_loss = tf.reduce_mean(
            tf.square(
                tf.nn.relu(
                    (margin_target - margin)
                    / tf.cast(np.deg2rad(0.5), prediction.dtype)
                )
            )
        )
        total = (
            beta_loss
            + tf.cast(lambda_fk, prediction.dtype) * fk_loss
            + tf.cast(lambda_margin, prediction.dtype) * margin_loss
        )
        return total, beta_loss, fk_loss, margin_loss

    def continuity_loss(training: bool) -> Any:
        prediction = model(xyz, training=training)
        loops = tf.reshape(
            prediction, (family_count, phase_count, 6)
        )
        rolled = tf.roll(loops, shift=-1, axis=1)
        edge_deg = tf.sqrt(
            tf.reduce_mean(tf.square(rolled - loops), axis=2)
        ) * tf.cast(180.0 / np.pi, prediction.dtype)
        return tf.reduce_mean(
            tf.square(
                tf.nn.relu(
                    (edge_deg - transition_target)
                    / tf.cast(0.25, prediction.dtype)
                )
            )
        )

    best_loss = np.inf
    best_weights = model.get_weights()
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(int(max_epochs)):
        for batch_xyz, batch_beta in dataset:
            with tf.GradientTape() as tape:
                total, _, _, _ = point_loss(
                    batch_xyz, batch_beta, True
                )
            gradients = tape.gradient(total, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
        if float(lambda_continuity) > 0.0:
            with tf.GradientTape() as tape:
                cyclic = continuity_loss(True)
                cyclic_weighted = (
                    tf.cast(lambda_continuity, cyclic.dtype) * cyclic
                )
            gradients = tape.gradient(
                cyclic_weighted, model.trainable_variables
            )
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )

        total, beta_loss, fk_loss, margin_loss = point_loss(
            xyz, beta, False
        )
        cyclic = continuity_loss(False)
        objective = total + tf.cast(
            lambda_continuity, total.dtype
        ) * cyclic
        row = {
            "epoch": int(epoch + 1),
            "objective": float(objective.numpy()),
            "beta_loss": float(beta_loss.numpy()),
            "fk_loss": float(fk_loss.numpy()),
            "margin_loss": float(margin_loss.numpy()),
            "continuity_loss": float(cyclic.numpy()),
        }
        history_rows.append(row)
        if row["objective"] < best_loss - 1.0e-7:
            best_loss = row["objective"]
            best_weights = model.get_weights()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(patience):
            break

    model.set_weights(best_weights)
    history = pd.DataFrame(history_rows)
    report = {
        "seed": int(seed),
        "row_count": int(len(frame)),
        "family_count": family_count,
        "phase_count": phase_count,
        "epoch_count": int(len(history)),
        "best_objective": float(best_loss),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "lambda_fk": float(lambda_fk),
        "lambda_margin": float(lambda_margin),
        "lambda_continuity": float(lambda_continuity),
        "margin_target_deg": float(margin_target_deg),
        "transition_target_deg": float(transition_target_deg),
        "beta_loss_scale_deg": list(map(float, beta_loss_scale_deg)),
    }
    return model, history, report


def save_uncompiled_model(model: Any, path: str | Path) -> None:
    """Persist a custom-loop model for later compile-free inference."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)
