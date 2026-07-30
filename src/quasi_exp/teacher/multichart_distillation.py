"""Known-chart single-model distillation for BACRA V12.16B."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .bacra_exploratory_expansion import point_margin_deg
from .dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from .region_growth import JACOBIAN_COLUMNS
from .retention_distillation import DISTILL_BETA_COLUMNS
from .student_tracking_tf import StudentGeometry


def build_chart_conditioned_model(source_model: Any) -> Any:
    """Expand one xyz Student to xyz + known-chart input without drift.

    The chart feature enters as a zero-initialized fourth row of the first
    Dense kernel.  Consequently chart feature 0 reproduces the source model
    exactly before fine-tuning, while one shared MLP can learn chart-B offsets.
    """

    import tensorflow as tf

    xyz = tf.keras.Input(shape=(3,), name="target_xyz_m")
    chart = tf.keras.Input(shape=(1,), name="chart_feature")
    source_normalization = source_model.get_layer("xyz_normalization")
    normalization = tf.keras.layers.Normalization(
        axis=-1, name="xyz_normalization"
    )
    normalized = normalization(xyz)
    combined = tf.keras.layers.Concatenate(name="xyz_chart_concat")(
        [normalized, chart]
    )
    dense_layers = [
        source_model.get_layer("dense_0"),
        source_model.get_layer("dense_1"),
        source_model.get_layer("dense_2"),
        source_model.get_layer("beta_latent"),
    ]
    hidden = combined
    cloned = []
    for source in dense_layers:
        layer = tf.keras.layers.Dense(
            int(source.units),
            activation=source.activation,
            use_bias=bool(source.use_bias),
            name=source.name,
        )
        hidden = layer(hidden)
        cloned.append(layer)
    unit = tf.keras.layers.Activation("tanh", name="beta_unit")(hidden)
    source_scale = source_model.get_layer("beta_rad")
    output = tf.keras.layers.Rescaling(
        scale=np.asarray(source_scale.scale, dtype=np.float32),
        offset=np.asarray(source_scale.offset, dtype=np.float32),
        name="beta_rad",
    )(unit)
    model = tf.keras.Model(
        inputs=[xyz, chart],
        outputs=output,
        name="known_chart_single_student",
    )
    normalization.set_weights(source_normalization.get_weights())
    normalization.finalize_state()
    for index, (source, target) in enumerate(zip(dense_layers, cloned, strict=True)):
        weights = source.get_weights()
        if index == 0:
            kernel, bias = weights
            expanded = np.vstack(
                [kernel, np.zeros((1, kernel.shape[1]), dtype=kernel.dtype)]
            )
            target.set_weights([expanded, bias])
        else:
            target.set_weights(weights)
    return model


def predict_chart_conditioned(
    model: Any,
    xyz_m: np.ndarray,
    chart_feature: np.ndarray | float,
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    xyz = np.asarray(xyz_m, dtype=np.float32).reshape(-1, 3)
    chart = np.asarray(chart_feature, dtype=np.float32)
    if chart.ndim == 0:
        chart = np.full((len(xyz), 1), float(chart), dtype=np.float32)
    else:
        chart = chart.reshape(-1, 1)
    if len(chart) != len(xyz):
        raise ValueError("chart feature length must match xyz rows")
    return np.asarray(
        model.predict([xyz, chart], batch_size=int(batch_size), verbose=0),
        dtype=float,
    )


def train_chart_conditioned_student(
    model: Any,
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    geometry: StudentGeometry,
    seed: int,
    learning_rate: float,
    batch_size: int,
    max_steps: int,
    validation_interval: int,
    patience_intervals: int,
    chart_sampling_weights: Mapping[str, float],
    teacher_beta_weight: float,
    distill_beta_weight: float,
    fk_weight: float,
    margin_weight: float,
    row_loss_weight: float,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Fine-tune one bounded MLP across explicitly registered charts."""

    import tensorflow as tf
    from quasi_exp.model.kinematics_tf import forward_xyz_from_beta_tf

    required = {
        *XYZ_COLUMNS,
        *BETA_COLUMNS,
        *DISTILL_BETA_COLUMNS,
        *JACOBIAN_COLUMNS,
        "chart_id",
        "chart_feature",
        "sample_weight",
    }
    for name, frame in (
        ("training", training_frame),
        ("validation", validation_frame),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} frame missing columns: {missing}")
    observed = sorted(map(str, training_frame["chart_id"].unique()))
    registered = {
        str(key): float(value)
        for key, value in chart_sampling_weights.items()
    }
    weights = np.asarray([registered[name] for name in observed], dtype=float)
    if (
        set(observed) != set(registered)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError(
            "chart sampling weights must cover observed charts and sum to one"
        )
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
        return (
            frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32),
            frame["chart_feature"].to_numpy(dtype=np.float32).reshape(-1, 1),
            frame.loc[:, BETA_COLUMNS].to_numpy(dtype=np.float32),
            frame.loc[:, DISTILL_BETA_COLUMNS].to_numpy(dtype=np.float32),
            frame["sample_weight"].to_numpy(dtype=np.float32),
            frame.loc[:, JACOBIAN_COLUMNS]
            .to_numpy(dtype=np.float32)
            .reshape(-1, 3, 6),
        )

    train_arrays = arrays(training_frame)
    valid_arrays = arrays(validation_frame)
    sources = []
    for offset, chart_id in enumerate(observed):
        selected = training_frame["chart_id"].eq(chart_id).to_numpy()
        chart_arrays = tuple(value[selected] for value in train_arrays)
        sources.append(
            tf.data.Dataset.from_tensor_slices(chart_arrays)
            .shuffle(
                int(np.count_nonzero(selected)),
                seed=int(seed) + offset,
                reshuffle_each_iteration=True,
            )
            .repeat()
        )
    dataset = tf.data.Dataset.sample_from_datasets(
        sources,
        weights=weights.tolist(),
        seed=int(seed),
        stop_on_empty_dataset=False,
    )
    iterator = iter(
        dataset.batch(int(batch_size), drop_remainder=False).prefetch(
            tf.data.AUTOTUNE
        )
    )
    bounds = tf.constant(
        np.asarray(geometry.beta_bounds_rad, dtype=np.float32),
        dtype=tf.float32,
    )
    beta_scale = tf.constant(
        np.deg2rad(np.asarray([1, 1, 1, 1, 0.75, 1], dtype=np.float32))
    )
    optimizer = tf.keras.optimizers.Adam(float(learning_rate))

    def loss_terms(batch: tuple[Any, ...], training: bool) -> tuple[Any, ...]:
        xyz, chart, beta_teacher, beta_distill, sample_weight, jac = batch
        prediction = model([xyz, chart], training=training)
        teacher_point = tf.reduce_mean(
            tf.square((prediction - beta_teacher) / beta_scale), axis=1
        )
        distill_point = tf.reduce_mean(
            tf.square((prediction - beta_distill) / beta_scale), axis=1
        )
        predicted_xyz = forward_xyz_from_beta_tf(
            prediction,
            lengths_m=geometry.lengths_m,
            p_end_local_m=geometry.p_end_local_m,
            theta_sign=geometry.theta_sign,
        )
        fk_point = tf.reduce_mean(
            tf.square(
                (predicted_xyz - xyz)
                / tf.cast(0.003, prediction.dtype)
            ),
            axis=1,
        )
        margin = tf.minimum(
            prediction - bounds[:, 0], bounds[:, 1] - prediction
        )
        margin_point = tf.reduce_mean(
            tf.square(
                tf.nn.relu(
                    (
                        tf.cast(np.deg2rad(1.5), prediction.dtype)
                        - margin
                    )
                    / tf.cast(np.deg2rad(0.5), prediction.dtype)
                )
            ),
            axis=1,
        )
        projected = tf.einsum(
            "nij,nj->ni", jac, prediction - beta_teacher
        )
        row_point = tf.reduce_mean(
            tf.square(
                projected / tf.cast(0.003, prediction.dtype)
            ),
            axis=1,
        )
        point = (
            tf.cast(teacher_beta_weight, prediction.dtype) * teacher_point
            + tf.cast(distill_beta_weight, prediction.dtype) * distill_point
            + tf.cast(fk_weight, prediction.dtype) * fk_point
            + tf.cast(margin_weight, prediction.dtype) * margin_point
            + tf.cast(row_loss_weight, prediction.dtype) * row_point
        )
        objective = tf.reduce_sum(point * sample_weight) / tf.reduce_sum(
            sample_weight
        )
        return (
            objective,
            tf.reduce_mean(teacher_point),
            tf.reduce_mean(distill_point),
            tf.reduce_mean(fk_point),
            tf.reduce_mean(margin_point),
            tf.reduce_mean(row_point),
        )

    best = math.inf
    best_weights = model.get_weights()
    stale = 0
    history_rows: list[dict[str, Any]] = []
    valid_tensor = tuple(
        tf.convert_to_tensor(value) for value in valid_arrays
    )
    for step in range(1, int(max_steps) + 1):
        batch = next(iterator)
        with tf.GradientTape() as tape:
            objective, *_rest = loss_terms(batch, True)
        gradients = tape.gradient(objective, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if (
            step % int(validation_interval) != 0
            and step != int(max_steps)
        ):
            continue
        terms = loss_terms(valid_tensor, False)
        value = float(terms[0].numpy())
        history_rows.append(
            {
                "step": int(step),
                "validation_objective": value,
                "validation_teacher_beta_loss": float(terms[1].numpy()),
                "validation_distill_beta_loss": float(terms[2].numpy()),
                "validation_fk_loss": float(terms[3].numpy()),
                "validation_margin_loss": float(terms[4].numpy()),
                "validation_row_loss": float(terms[5].numpy()),
            }
        )
        if value < best - 1.0e-7:
            best = value
            best_weights = model.get_weights()
            stale = 0
        else:
            stale += 1
            if stale >= int(patience_intervals):
                break
    model.set_weights(best_weights)
    validation_prediction = predict_chart_conditioned(
        model,
        validation_frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
        validation_frame["chart_feature"].to_numpy(dtype=float),
    )
    report = {
        "model_mode": "known_chart_single_bounded_mlp",
        "seed": int(seed),
        "optimizer_steps_completed": int(
            history_rows[-1]["step"] if history_rows else 0
        ),
        "best_validation_objective": float(best),
        "chart_sampling_weights": registered,
        "training_chart_rows": {
            chart_id: int(
                training_frame["chart_id"].eq(chart_id).sum()
            )
            for chart_id in observed
        },
        "validation_chart_rows": {
            chart_id: int(
                validation_frame["chart_id"].eq(chart_id).sum()
            )
            for chart_id in observed
        },
        "validation_minimum_joint_margin_deg": float(
            point_margin_deg(validation_prediction, geometry.beta_bounds_rad).min()
        ),
    }
    return model, pd.DataFrame(history_rows), report
