"""Bounded parity-smooth G4 Student and retry18 training helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS
from .region_growth import JACOBIAN_COLUMNS
from .student_tracking_tf import StudentGeometry


@dataclass(frozen=True)
class ParityStudentConfig:
    hidden_units: tuple[int, ...] = (128, 128, 64)
    learning_rate: float = 1.0e-3
    maximum_steps: int = 1500
    validation_interval: int = 50
    patience_intervals: int = 8
    seed: int = 20260925
    axial_scale_mm: float = 600.0
    radial_scale_mm: float = 200.0
    jacobian_lambda: float = 0.0

    def __post_init__(self) -> None:
        if not self.hidden_units or any(int(value) <= 0 for value in self.hidden_units):
            raise ValueError("parity hidden_units must be positive")
        numeric = (
            self.learning_rate,
            self.axial_scale_mm,
            self.radial_scale_mm,
            self.jacobian_lambda,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("parity Student numeric config must be finite")
        if self.learning_rate <= 0 or self.axial_scale_mm <= 0 or self.radial_scale_mm <= 0:
            raise ValueError("parity Student scales and learning rate must be positive")
        if self.jacobian_lambda < 0:
            raise ValueError("jacobian_lambda must be non-negative")


def _keras_layers() -> tuple[Any, Any]:
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="quasi_exp")
    class ParityFeatures(tf.keras.layers.Layer):
        def __init__(self, zero_x_m: float, axial_scale_mm: float, radial_scale_mm: float, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.zero_x_m = float(zero_x_m)
            self.axial_scale_mm = float(axial_scale_mm)
            self.radial_scale_mm = float(radial_scale_mm)

        def call(self, xyz: Any) -> Any:
            xyz = tf.convert_to_tensor(xyz)
            u = (tf.cast(self.zero_x_m, xyz.dtype) - xyz[:, 0]) * tf.cast(1000.0 / self.axial_scale_mm, xyz.dtype)
            y = xyz[:, 1] * tf.cast(1000.0 / self.radial_scale_mm, xyz.dtype)
            z = xyz[:, 2] * tf.cast(1000.0 / self.radial_scale_mm, xyz.dtype)
            return tf.stack((u, tf.square(y), tf.square(z)), axis=1)

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "zero_x_m": self.zero_x_m, "axial_scale_mm": self.axial_scale_mm, "radial_scale_mm": self.radial_scale_mm}

    @tf.keras.utils.register_keras_serializable(package="quasi_exp")
    class ParityProjection(tf.keras.layers.Layer):
        def __init__(self, radial_scale_mm: float, beta_halfspan_rad: Sequence[float], **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.radial_scale_mm = float(radial_scale_mm)
            self.beta_halfspan_rad = tuple(float(value) for value in beta_halfspan_rad)

        def call(self, values: Any) -> Any:
            xyz, latent = values
            xyz = tf.convert_to_tensor(xyz)
            latent = tf.convert_to_tensor(latent)
            y = xyz[:, 1:2] * tf.cast(1000.0 / self.radial_scale_mm, latent.dtype)
            z = xyz[:, 2:3] * tf.cast(1000.0 / self.radial_scale_mm, latent.dtype)
            parity = tf.concat((y, z, y, z, y, z), axis=1)
            halfspan = tf.constant(self.beta_halfspan_rad, dtype=latent.dtype)
            return tf.math.tanh(parity * latent) * halfspan

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "radial_scale_mm": self.radial_scale_mm, "beta_halfspan_rad": self.beta_halfspan_rad}

    return ParityFeatures, ParityProjection


def build_parity_smooth_student(
    *,
    train_xyz_m: np.ndarray,
    zero_x_m: float,
    geometry: StudentGeometry,
    config: ParityStudentConfig,
) -> Any:
    import tensorflow as tf

    bounds = np.asarray(geometry.beta_bounds_rad, dtype=np.float32)
    if not np.allclose(bounds[:, 0], -bounds[:, 1], atol=1.0e-8, rtol=0):
        raise ValueError("parity Student requires symmetric beta bounds")
    ParityFeatures, ParityProjection = _keras_layers()
    inputs = tf.keras.Input((3,), name="xyz_m")
    features = ParityFeatures(zero_x_m, config.axial_scale_mm, config.radial_scale_mm, name="parity_features")(inputs)
    normalization = tf.keras.layers.Normalization(name="feature_normalization")
    raw_features_model = tf.keras.Model(inputs, features)
    normalization.adapt(np.asarray(raw_features_model(np.asarray(train_xyz_m, dtype=np.float32)), dtype=np.float32))
    hidden: Any = normalization(features)
    for index, units in enumerate(config.hidden_units):
        hidden = tf.keras.layers.Dense(int(units), activation="gelu", name=f"dense_{index}")(hidden)
    latent = tf.keras.layers.Dense(6, name="parity_latent")(hidden)
    output = ParityProjection(config.radial_scale_mm, bounds[:, 1], name="beta_rad")((inputs, latent))
    return tf.keras.Model(inputs, output, name="retry18_parity_smooth_student")


def parity_loss_terms(
    *,
    model: Any,
    xyz_m: Any,
    beta_true: Any,
    jacobian_true: Any,
    sample_weight: Any,
    beta_coordinate_weights: Sequence[float],
    jacobian_lambda: float,
    training: bool,
) -> Mapping[str, Any]:
    import tensorflow as tf

    prediction = model(xyz_m, training=training)
    beta_true = tf.convert_to_tensor(beta_true, dtype=prediction.dtype)
    jacobian = tf.convert_to_tensor(jacobian_true, dtype=prediction.dtype)
    sample_weight = tf.reshape(tf.convert_to_tensor(sample_weight, dtype=prediction.dtype), (-1,))
    coordinate_weights = tf.constant(tuple(float(value) for value in beta_coordinate_weights), dtype=prediction.dtype)
    squared = tf.square(coordinate_weights)
    beta_point = tf.reduce_sum(squared * tf.square(prediction - beta_true), axis=1) / tf.reduce_sum(squared)
    projected = tf.einsum("nij,nj->ni", jacobian, prediction - beta_true)
    row_point = tf.reduce_mean(tf.square(projected / tf.cast(0.003, prediction.dtype)), axis=1)
    point = beta_point + tf.cast(float(jacobian_lambda), prediction.dtype) * row_point
    denominator = tf.reduce_sum(sample_weight)
    return {
        "objective": tf.reduce_sum(point * sample_weight) / denominator,
        "beta": tf.reduce_sum(beta_point * sample_weight) / denominator,
        "row_space": tf.reduce_sum(row_point * sample_weight) / denominator,
        "prediction": prediction,
    }


def _arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return (
        frame.loc[:, XYZ_COLUMNS].to_numpy(np.float32),
        frame.loc[:, BETA_COLUMNS].to_numpy(np.float32),
        frame.loc[:, JACOBIAN_COLUMNS].to_numpy(np.float32).reshape(-1, 3, 6),
        frame["sample_weight"].to_numpy(np.float32),
    )


def train_parity_smooth_student(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    zero_x_m: float,
    geometry: StudentGeometry,
    config: ParityStudentConfig,
    beta_coordinate_weights: Sequence[float] = (4, 4, 2, 2, 1, 1),
) -> tuple[Any, pd.DataFrame]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(config.seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    train_arrays = tuple(tf.convert_to_tensor(value) for value in _arrays(train))
    validation_arrays = tuple(tf.convert_to_tensor(value) for value in _arrays(validation))
    model = build_parity_smooth_student(train_xyz_m=train_arrays[0], zero_x_m=zero_x_m, geometry=geometry, config=config)
    optimizer = tf.keras.optimizers.Adam(float(config.learning_rate))
    best = math.inf
    best_weights = model.get_weights()
    stale = 0
    rows: list[dict[str, float | int]] = []
    for step in range(1, int(config.maximum_steps) + 1):
        with tf.GradientTape() as tape:
            terms = parity_loss_terms(model=model, xyz_m=train_arrays[0], beta_true=train_arrays[1], jacobian_true=train_arrays[2], sample_weight=train_arrays[3], beta_coordinate_weights=beta_coordinate_weights, jacobian_lambda=config.jacobian_lambda, training=True)
        gradients = tape.gradient(terms["objective"], model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if step % int(config.validation_interval) and step != int(config.maximum_steps):
            continue
        valid = parity_loss_terms(model=model, xyz_m=validation_arrays[0], beta_true=validation_arrays[1], jacobian_true=validation_arrays[2], sample_weight=validation_arrays[3], beta_coordinate_weights=beta_coordinate_weights, jacobian_lambda=config.jacobian_lambda, training=False)
        value = float(valid["objective"].numpy())
        rows.append({"step": step, "validation_objective": value, "validation_beta_loss": float(valid["beta"].numpy()), "validation_row_space_loss": float(valid["row_space"].numpy())})
        if value < best - 1.0e-7:
            best = value
            best_weights = model.get_weights()
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience_intervals):
            break
    model.set_weights(best_weights)
    return model, pd.DataFrame(rows)


def mirror_beta(beta: np.ndarray, *, mirror_y: bool = False, mirror_z: bool = False) -> np.ndarray:
    value = np.asarray(beta, dtype=float).copy()
    if mirror_y:
        value[..., [0, 2, 4]] *= -1.0
    if mirror_z:
        value[..., [1, 3, 5]] *= -1.0
    return value


def seam_boundary_metrics(model: Any, xyz_m: np.ndarray, *, seam: str, epsilon_mm: float = 1.0) -> Mapping[str, float]:
    points = np.asarray(xyz_m, dtype=float).reshape(-1, 3)
    if seam not in {"y", "z"}:
        raise ValueError("seam must be y or z")
    axis = 1 if seam == "y" else 2
    odd = [0, 2, 4] if seam == "y" else [1, 3, 5]
    even = [1, 3, 5] if seam == "y" else [0, 2, 4]
    center = points.copy(); center[:, axis] = 0.0
    positive = center.copy(); positive[:, axis] = float(epsilon_mm) / 1000.0
    negative = center.copy(); negative[:, axis] = -float(epsilon_mm) / 1000.0
    beta0 = np.asarray(model(center.astype(np.float32), training=False), dtype=float)
    beta_plus = np.asarray(model(positive.astype(np.float32), training=False), dtype=float)
    beta_minus = np.asarray(model(negative.astype(np.float32), training=False), dtype=float)
    odd_zero = np.sqrt(np.mean(np.square(np.rad2deg(beta0[:, odd])), axis=1))
    even_slope = np.sqrt(np.mean(np.square(np.rad2deg(beta_plus[:, even] - beta_minus[:, even])), axis=1)) / (2.0 * float(epsilon_mm))
    jump = np.sqrt(np.mean(np.square(np.rad2deg(beta_plus - beta_minus)), axis=1))
    return {"odd_zero_maximum_deg": float(np.max(odd_zero)), "even_normal_slope_maximum_deg_per_mm": float(np.max(even_slope)), "cross_seam_beta_rms_maximum_deg": float(np.max(jump))}


def save_parity_student(model: Any, path: str | Path) -> None:
    model.save(Path(path))


def data_driven_radial_scale_mm(
    frame: pd.DataFrame,
    *,
    quantile: float = 0.99,
    safety_factor: float = 1.05,
) -> float:
    """Freeze the parity scale from the unified supervision denominator."""

    if not 0.0 < float(quantile) <= 1.0 or float(safety_factor) < 1.0:
        raise ValueError("invalid parity radial-scale policy")
    rho_mm = 1000.0 * np.hypot(frame["y_m"].to_numpy(float), frame["z_m"].to_numpy(float))
    if not len(rho_mm) or not np.isfinite(rho_mm).all():
        raise ValueError("parity radial scale requires finite supervision rows")
    return max(1.0, float(safety_factor) * float(np.quantile(rho_mm, float(quantile))))


def load_parity_student(path: str | Path) -> Any:
    import tensorflow as tf

    ParityFeatures, ParityProjection = _keras_layers()
    return tf.keras.models.load_model(Path(path), compile=False, custom_objects={"ParityFeatures": ParityFeatures, "ParityProjection": ParityProjection})
