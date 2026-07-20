"""TensorFlow students and leakage-safe autoregressive rollout for V10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from quasi_exp.model.kinematics_tf import forward_xyz_from_beta_tf
from quasi_exp.teacher.student import BETA_COLUMNS, XYZ_COLUMNS


OutputMode = Literal["identity", "tanh"]


@dataclass(frozen=True)
class StudentGeometry:
    lengths_m: np.ndarray
    p_end_local_m: np.ndarray
    theta_sign: float
    beta_bounds_rad: np.ndarray

    def __post_init__(self) -> None:
        lengths = np.asarray(self.lengths_m, dtype=np.float32).reshape(31)
        endpoint = np.asarray(self.p_end_local_m, dtype=np.float32).reshape(4)
        bounds = np.asarray(self.beta_bounds_rad, dtype=np.float32).reshape(6, 2)
        if not np.isfinite(lengths).all() or not np.isfinite(endpoint).all():
            raise ValueError("student geometry must be finite")
        if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("student beta bounds must be finite and ordered")
        object.__setattr__(self, "lengths_m", lengths)
        object.__setattr__(self, "p_end_local_m", endpoint)
        object.__setattr__(self, "beta_bounds_rad", bounds)


def decode_beta_tf(latent: Any, bounds_rad: np.ndarray, mode: OutputMode) -> Any:
    import tensorflow as tf

    if mode == "identity":
        return tf.identity(latent)
    if mode != "tanh":
        raise ValueError(f"unsupported beta output mode: {mode}")
    bounds = tf.constant(np.asarray(bounds_rad, dtype=np.float32), dtype=latent.dtype)
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    halfspan = 0.5 * (bounds[:, 1] - bounds[:, 0])
    return midpoint + halfspan * tf.tanh(latent)


def student_loss_terms(
    y_true_packed: Any,
    beta_pred: Any,
    *,
    geometry: StudentGeometry,
    lambda_fk: float,
) -> dict[str, Any]:
    """Return normalized beta and millimetre-scale differentiable FK losses."""

    import tensorflow as tf

    packed = tf.convert_to_tensor(y_true_packed, dtype=beta_pred.dtype)
    beta_true = packed[..., :6]
    xyz_true = packed[..., 6:9]
    bounds = tf.constant(geometry.beta_bounds_rad, dtype=beta_pred.dtype)
    span = tf.maximum(bounds[:, 1] - bounds[:, 0], tf.cast(1.0e-6, beta_pred.dtype))
    beta_delta = (beta_pred - beta_true) / span
    beta_loss = tf.reduce_mean(tf.keras.losses.huber(beta_delta, tf.zeros_like(beta_delta)))
    if float(lambda_fk) == 0.0:
        zero = tf.zeros((), dtype=beta_pred.dtype)
        return {"total": beta_loss, "beta": beta_loss, "fk": zero}
    flat_beta = tf.reshape(beta_pred, (-1, 6))
    xyz_pred = forward_xyz_from_beta_tf(
        flat_beta,
        lengths_m=geometry.lengths_m,
        p_end_local_m=geometry.p_end_local_m,
        theta_sign=geometry.theta_sign,
    )
    xyz_pred = tf.reshape(xyz_pred, tf.shape(xyz_true))
    residual_mm = (xyz_pred - xyz_true) * tf.cast(1000.0, beta_pred.dtype)
    # Scale by 3 mm, the registered label-eligibility and tracking boundary.
    fk_loss = tf.reduce_mean(tf.keras.losses.huber(residual_mm / 3.0, tf.zeros_like(residual_mm)))
    total = beta_loss + tf.cast(lambda_fk, beta_pred.dtype) * fk_loss
    return {"total": total, "beta": beta_loss, "fk": fk_loss}


def packed_targets(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame.loc[:, BETA_COLUMNS].to_numpy(dtype=np.float32),
            frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32),
        ]
    )


def build_static_model(
    train_xyz: np.ndarray,
    *,
    geometry: StudentGeometry,
    output_mode: OutputMode,
    hidden_units: tuple[int, ...] = (128, 128, 64),
) -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input((3,), name="target_xyz_m")
    normalization = tf.keras.layers.Normalization(name="xyz_normalization")
    normalization.adapt(np.asarray(train_xyz, dtype=np.float32))
    hidden = normalization(inputs)
    for index, units in enumerate(hidden_units):
        hidden = tf.keras.layers.Dense(int(units), activation="gelu", name=f"dense_{index}")(hidden)
    latent = tf.keras.layers.Dense(6, name="beta_latent")(hidden)
    beta = _keras_beta_output(latent, geometry=geometry, output_mode=output_mode)
    return tf.keras.Model(inputs, beta, name=f"static_student_{output_mode}")


def _keras_beta_output(latent: Any, *, geometry: StudentGeometry, output_mode: OutputMode) -> Any:
    import tensorflow as tf

    if output_mode == "identity":
        return latent
    bounds = np.asarray(geometry.beta_bounds_rad, dtype=np.float32)
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    halfspan = 0.5 * (bounds[:, 1] - bounds[:, 0])
    unit_beta = tf.keras.layers.Activation("tanh", name="beta_unit")(latent)
    return tf.keras.layers.Rescaling(halfspan, offset=midpoint, name="beta_rad")(unit_beta)


def compile_student(model: Any, *, geometry: StudentGeometry, lambda_fk: float, learning_rate: float) -> None:
    import tensorflow as tf

    def loss(y_true: Any, y_pred: Any) -> Any:
        return student_loss_terms(
            y_true,
            y_pred,
            geometry=geometry,
            lambda_fk=float(lambda_fk),
        )["total"]

    model.compile(optimizer=tf.keras.optimizers.Adam(float(learning_rate)), loss=loss)


def make_cyclic_windows(
    frame: pd.DataFrame,
    *,
    window_size: int,
    include_reverse: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create all cyclic cuts; labels are used only in teacher-forced fitting."""

    ordered = frame.sort_values("phase_idx", kind="stable")
    xyz = ordered.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    beta = ordered.loc[:, BETA_COLUMNS].to_numpy(dtype=np.float32)
    if len(xyz) < int(window_size):
        raise ValueError("cyclic sequence is shorter than window_size")
    feature_windows: list[np.ndarray] = []
    target_windows: list[np.ndarray] = []
    directions = (1, -1) if include_reverse else (1,)
    base = np.arange(len(xyz))
    for direction in directions:
        traversal = base if direction == 1 else base[::-1]
        for cut in range(len(xyz)):
            rolled = np.roll(traversal, -cut)
            indices = rolled[: int(window_size)]
            previous_indices = np.roll(rolled, 1)[: int(window_size)]
            step_xyz = xyz[indices]
            previous_xyz = xyz[previous_indices]
            previous_beta = beta[previous_indices]
            feature_windows.append(
                np.column_stack([step_xyz, step_xyz - previous_xyz, previous_beta])
            )
            target_windows.append(np.column_stack([beta[indices], step_xyz]))
    return np.asarray(feature_windows, dtype=np.float32), np.asarray(target_windows, dtype=np.float32)


def build_gru_model(
    train_features: np.ndarray,
    *,
    geometry: StudentGeometry,
    output_mode: OutputMode = "tanh",
    units: int = 64,
) -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input((None, 12), name="tracking_sequence")
    normalization = tf.keras.layers.Normalization(axis=-1, name="feature_normalization")
    normalization.adapt(np.asarray(train_features, dtype=np.float32).reshape(-1, 12))
    hidden = normalization(inputs)
    hidden = tf.keras.layers.GRU(int(units), return_sequences=True, name="gru_0")(hidden)
    hidden = tf.keras.layers.GRU(int(units), return_sequences=True, name="gru_1")(hidden)
    latent = tf.keras.layers.Dense(6, name="beta_latent")(hidden)
    beta = _keras_beta_output(latent, geometry=geometry, output_mode=output_mode)
    return tf.keras.Model(inputs, beta, name="windowed_autoregressive_gru_student")


def autoregressive_rollout(
    model: Any,
    target_xyz_m: np.ndarray,
    *,
    initial_beta_rad: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Roll a sequence without reading any teacher beta after initialisation."""

    target = np.asarray(target_xyz_m, dtype=np.float32).reshape(-1, 3)
    previous_beta = np.asarray(initial_beta_rad, dtype=np.float32).reshape(6)
    features: list[np.ndarray] = []
    prediction = np.empty((len(target), 6), dtype=np.float32)
    for index, xyz in enumerate(target):
        previous_xyz = target[index - 1] if index > 0 else target[-1]
        features.append(np.concatenate([xyz, xyz - previous_xyz, previous_beta]))
        window = np.asarray(features[-int(window_size) :], dtype=np.float32)[None, ...]
        output = model(window, training=False)
        current = np.asarray(output)[0, -1].reshape(6)
        if not np.isfinite(current).all():
            raise RuntimeError(f"student rollout became non-finite at phase {index}")
        prediction[index] = current
        previous_beta = current
    return prediction
