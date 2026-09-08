"""Direct signed Student contracts for retry19.

This module keeps the representation contrast auditable: Q0-augmented and
F0-direct are derived from the same ordered full-G4 rows, while only the
former canonicalizes signed rows back to the quotient.  It also owns the
finite-gradient signed-power feature, unified signed macroblock splits, and
Teacher-delta graph-edge loss.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .region_growth import JACOBIAN_COLUMNS
from .retry12_symmetry import (
    BETA_COLUMNS,
    SYMMETRY_BETA_SIGNS,
    XYZ_COLUMNS,
)


@dataclass(frozen=True)
class DirectStudentConfig:
    hidden_units: tuple[int, ...] = (128, 128, 64)
    learning_rate: float = 1.0e-3
    maximum_steps: int = 1500
    validation_interval: int = 50
    patience_intervals: int = 8
    batch_size: int = 1024
    seed: int = 20260925
    axial_scale_mm: float = 600.0
    radial_scale_mm: float = 220.0
    signed_power_alpha: float = 0.5
    signed_power_epsilon_mm: float = 3.0
    edge_lambda: float = 0.0
    jacobian_lambda: float = 0.0

    def __post_init__(self) -> None:
        if not self.hidden_units or any(int(value) <= 0 for value in self.hidden_units):
            raise ValueError("Direct Student hidden units must be positive")
        if min(self.learning_rate, self.axial_scale_mm, self.radial_scale_mm) <= 0:
            raise ValueError("Direct Student scales and learning rate must be positive")
        if self.maximum_steps < 1 or self.validation_interval < 1 or self.batch_size < 1:
            raise ValueError("Direct Student training schedule must be positive")
        if not 0 < self.signed_power_alpha <= 1:
            raise ValueError("signed-power alpha must be in (0, 1]")
        if self.signed_power_epsilon_mm <= 0:
            raise ValueError("signed-power epsilon must be positive")
        if self.edge_lambda < 0 or self.jacobian_lambda < 0:
            raise ValueError("regularization weights must be non-negative")


def smooth_signed_power(
    values: np.ndarray | Sequence[float],
    *,
    alpha: float,
    epsilon: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not 0 < float(alpha) <= 1 or float(epsilon) <= 0:
        raise ValueError("smooth signed-power requires alpha in (0,1] and epsilon > 0")
    exponent = 0.5 * (1.0 - float(alpha))
    return values / np.power(np.square(values) + float(epsilon) ** 2, exponent)


def smooth_signed_power_derivative(
    values: np.ndarray | Sequence[float],
    *,
    alpha: float,
    epsilon: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not 0 < float(alpha) <= 1 or float(epsilon) <= 0:
        raise ValueError("smooth signed-power requires alpha in (0,1] and epsilon > 0")
    square = np.square(values) + float(epsilon) ** 2
    exponent = 0.5 * (1.0 - float(alpha))
    return np.power(square, -exponent) * (
        1.0 - (1.0 - float(alpha)) * np.square(values) / square
    )


def direct_feature_matrix(
    xyz_m: np.ndarray,
    *,
    zero_x_m: float,
    axial_scale_mm: float,
    radial_scale_mm: float,
    signed_power_alpha: float | None,
    signed_power_epsilon_mm: float = 3.0,
) -> np.ndarray:
    xyz = np.asarray(xyz_m, dtype=float).reshape(-1, 3)
    raw = np.column_stack(
        [
            (float(zero_x_m) - xyz[:, 0]) * 1000.0 / float(axial_scale_mm),
            xyz[:, 1] * 1000.0 / float(radial_scale_mm),
            xyz[:, 2] * 1000.0 / float(radial_scale_mm),
        ]
    )
    if signed_power_alpha is None:
        return raw
    epsilon = float(signed_power_epsilon_mm) / float(radial_scale_mm)
    signed = smooth_signed_power(raw[:, 1:], alpha=float(signed_power_alpha), epsilon=epsilon)
    return np.column_stack([raw, signed])


def _canonical_element(y_m: float, z_m: float) -> str:
    if y_m < 0 and z_m < 0:
        return "rotate_x_180"
    if y_m < 0:
        return "mirror_y"
    if z_m < 0:
        return "mirror_z"
    return "identity"


def aligned_q0_augmented_and_f0_rows(full_g4_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a row-identical quotient/direct causal contrast.

    Row order, weights, batch ordinals, and control IDs remain identical.  Q0
    differs only by canonicalizing signed positions and beta labels.
    """

    required = {*XYZ_COLUMNS, *BETA_COLUMNS}
    if not required <= set(full_g4_rows):
        raise ValueError("full-G4 frame is missing xyz/beta columns")
    direct = full_g4_rows.copy().reset_index(drop=True)
    direct["control_row_id"] = direct.get(
        "control_row_id", pd.Series([f"control:{index:08d}" for index in range(len(direct))])
    )
    direct["control_row_ordinal"] = np.arange(len(direct), dtype=np.int64)
    direct["sample_weight"] = direct.get("sample_weight", 1.0)
    if "orbit_size" in direct:
        direct["orbit_normalized_weight"] = direct["sample_weight"].to_numpy(float) / direct[
            "orbit_size"
        ].to_numpy(float)
    else:
        direct["orbit_normalized_weight"] = direct["sample_weight"].to_numpy(float)

    quotient = direct.copy()
    beta = quotient.loc[:, BETA_COLUMNS].to_numpy(float)
    elements: list[str] = []
    for index, (y_m, z_m) in enumerate(quotient.loc[:, ["y_m", "z_m"]].to_numpy(float)):
        element = _canonical_element(float(y_m), float(z_m))
        elements.append(element)
        beta[index] *= SYMMETRY_BETA_SIGNS[element]
    quotient.loc[:, BETA_COLUMNS] = beta
    quotient["y_m"] = quotient["y_m"].abs()
    quotient["z_m"] = quotient["z_m"].abs()
    quotient["canonicalization_element"] = elements
    quotient["representation"] = "quotient_augmented"
    direct["canonicalization_element"] = "identity"
    direct["representation"] = "direct_signed"
    shared = ["control_row_id", "control_row_ordinal", "orbit_normalized_weight"]
    if not quotient.loc[:, shared].equals(direct.loc[:, shared]):
        raise AssertionError("Q0-augmented and F0-direct controls lost row alignment")
    return quotient, direct


def signed_macroblock_ids(frame: pd.DataFrame, *, block_size_mm: float = 40.0) -> pd.Series:
    if not set(XYZ_COLUMNS) <= set(frame):
        raise ValueError("signed macroblocks require xyz columns")
    bins = np.floor(frame.loc[:, XYZ_COLUMNS].to_numpy(float) * 1000.0 / float(block_size_mm)).astype(np.int64)
    return pd.Series([f"smacro:{x}:{y}:{z}" for x, y, z in bins], index=frame.index, dtype=str)


def unified_split_registry(
    rows: pd.DataFrame,
    historical_rows: pd.DataFrame,
    *,
    block_size_mm: float = 40.0,
    seed: int = 20260950,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign one split per signed macroblock across historical sources."""

    current = rows.copy()
    current["macroblock_id"] = signed_macroblock_ids(current, block_size_mm=block_size_mm)
    history = historical_rows.copy()
    if len(history):
        history["macroblock_id"] = signed_macroblock_ids(history, block_size_mm=block_size_mm)
        split_column = "split_role" if "split_role" in history else "split"
        if split_column not in history:
            raise ValueError("historical rows require split_role or split")
        role_counts = history.groupby("macroblock_id")[split_column].nunique()
        conflicting_blocks = set(role_counts[role_counts.gt(1)].index.astype(str))
        stable_history = history[~history["macroblock_id"].astype(str).isin(conflicting_blocks)]
        inherited = stable_history.groupby("macroblock_id", sort=True)[split_column].first().astype(str).to_dict()
    else:
        conflicting_blocks = set()
        inherited = {}
    registry_rows: list[dict[str, Any]] = []
    for block in sorted(current["macroblock_id"].astype(str).unique()):
        if block in inherited:
            split = inherited[block]
            reason = "inherited_retry18_macroblock"
        else:
            value = int(hashlib.sha256(f"{seed}:{block}".encode()).hexdigest(), 16) / float(2**256)
            split = "train" if value < 0.70 else "validation" if value < 0.85 else "test"
            reason = (
                "retry18_conflict_reassigned_by_retry19_seed_hash"
                if block in conflicting_blocks
                else "retry19_seed_hash"
            )
        registry_rows.append(
            {
                "macroblock_id": block,
                "split_role": split,
                "assignment_reason": reason,
                "split_seed": int(seed),
                "block_size_mm": float(block_size_mm),
            }
        )
    registry = pd.DataFrame(registry_rows)
    split_by_block = registry.set_index("macroblock_id")["split_role"].to_dict()
    current["split_role"] = current["macroblock_id"].map(split_by_block)
    audit_only = current.get("domain_class", pd.Series("primary", index=current.index)).astype(str).eq("audit_only")
    current.loc[audit_only, "split_role"] = "audit_only"
    return current, registry


def edge_delta_loss_numpy(
    predicted_beta: np.ndarray,
    teacher_beta: np.ndarray,
    xyz_m: np.ndarray,
    edge_indices: np.ndarray,
    *,
    edge_weights: np.ndarray | None = None,
    beta_coordinate_weights: Sequence[float] = (4, 4, 2, 2, 1, 1),
    minimum_distance_mm: float = 5.0,
) -> float:
    prediction = np.asarray(predicted_beta, dtype=float).reshape(-1, 6)
    teacher = np.asarray(teacher_beta, dtype=float).reshape(-1, 6)
    xyz = np.asarray(xyz_m, dtype=float).reshape(-1, 3)
    edges = np.asarray(edge_indices, dtype=int).reshape(-1, 2)
    if len(edges) == 0:
        return 0.0
    if min(edges.ravel()) < 0 or max(edges.ravel()) >= len(prediction):
        raise ValueError("edge index is outside the supervision frame")
    delta_error = (prediction[edges[:, 0]] - prediction[edges[:, 1]]) - (
        teacher[edges[:, 0]] - teacher[edges[:, 1]]
    )
    distance_mm = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1) * 1000.0
    rate = delta_error / np.maximum(distance_mm, float(minimum_distance_mm))[:, None]
    coordinate = np.square(np.asarray(beta_coordinate_weights, dtype=float).reshape(1, 6))
    point = np.sum(coordinate * np.square(rate), axis=1)
    weight = np.ones(len(edges), dtype=float) if edge_weights is None else np.asarray(edge_weights, dtype=float)
    return float(np.sum(weight * point) / np.sum(weight))


def _keras_direct_feature_layer() -> Any:
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="quasi_exp")
    class DirectSignedFeatures(tf.keras.layers.Layer):
        def __init__(
            self,
            zero_x_m: float,
            axial_scale_mm: float,
            radial_scale_mm: float,
            signed_power_alpha: float | None,
            signed_power_epsilon_mm: float,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.zero_x_m = float(zero_x_m)
            self.axial_scale_mm = float(axial_scale_mm)
            self.radial_scale_mm = float(radial_scale_mm)
            self.signed_power_alpha = None if signed_power_alpha is None else float(signed_power_alpha)
            self.signed_power_epsilon_mm = float(signed_power_epsilon_mm)

        def call(self, xyz: Any) -> Any:
            xyz = tf.convert_to_tensor(xyz)
            dtype = xyz.dtype
            u = (tf.cast(self.zero_x_m, dtype) - xyz[:, 0]) * tf.cast(1000.0 / self.axial_scale_mm, dtype)
            y = xyz[:, 1] * tf.cast(1000.0 / self.radial_scale_mm, dtype)
            z = xyz[:, 2] * tf.cast(1000.0 / self.radial_scale_mm, dtype)
            raw = tf.stack((u, y, z), axis=1)
            if self.signed_power_alpha is None:
                return raw
            radial = tf.stack((y, z), axis=1)
            epsilon = tf.cast(self.signed_power_epsilon_mm / self.radial_scale_mm, dtype)
            exponent = tf.cast(0.5 * (1.0 - self.signed_power_alpha), dtype)
            signed = radial / tf.pow(tf.square(radial) + tf.square(epsilon), exponent)
            return tf.concat((raw, signed), axis=1)

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "zero_x_m": self.zero_x_m,
                "axial_scale_mm": self.axial_scale_mm,
                "radial_scale_mm": self.radial_scale_mm,
                "signed_power_alpha": self.signed_power_alpha,
                "signed_power_epsilon_mm": self.signed_power_epsilon_mm,
            }

    return DirectSignedFeatures


def build_direct_student(
    *,
    train_xyz_m: np.ndarray,
    zero_x_m: float,
    beta_bounds_rad: np.ndarray,
    config: DirectStudentConfig,
    include_signed_power: bool,
) -> Any:
    import tensorflow as tf

    bounds = np.asarray(beta_bounds_rad, dtype=np.float32).reshape(6, 2)
    centre = np.mean(bounds, axis=1)
    halfspan = 0.5 * (bounds[:, 1] - bounds[:, 0])
    DirectSignedFeatures = _keras_direct_feature_layer()
    inputs = tf.keras.Input((3,), name="xyz_m")
    features = DirectSignedFeatures(
        zero_x_m,
        config.axial_scale_mm,
        config.radial_scale_mm,
        config.signed_power_alpha if include_signed_power else None,
        config.signed_power_epsilon_mm,
        name="direct_signed_features",
    )(inputs)
    feature_model = tf.keras.Model(inputs, features)
    normalization = tf.keras.layers.Normalization(name="feature_normalization")
    normalization.adapt(np.asarray(feature_model(np.asarray(train_xyz_m, np.float32)), np.float32))
    hidden: Any = normalization(features)
    for index, units in enumerate(config.hidden_units):
        hidden = tf.keras.layers.Dense(int(units), activation="gelu", name=f"dense_{index}")(hidden)
    latent = tf.keras.layers.Dense(6, activation="tanh", name="beta_latent")(hidden)
    output = tf.keras.layers.Rescaling(scale=halfspan, offset=centre, name="beta_rad")(latent)
    return tf.keras.Model(inputs, output, name="retry19_direct_signed_student")


def load_direct_student(path: Any) -> Any:
    """Load a saved Direct Student after registering its feature layer."""

    import tensorflow as tf

    _keras_direct_feature_layer()
    return tf.keras.models.load_model(path, compile=False)


def train_direct_student(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    train_edges: pd.DataFrame,
    *,
    zero_x_m: float,
    beta_bounds_rad: np.ndarray,
    config: DirectStudentConfig,
    include_signed_power: bool,
    beta_coordinate_weights: Sequence[float] = (4, 4, 2, 2, 1, 1),
) -> tuple[Any, pd.DataFrame]:
    """Deterministic mini-batch trainer with Teacher-delta edge loss."""

    import tensorflow as tf

    if training.empty or validation.empty:
        raise ValueError("Direct Student requires non-empty train and validation frames")
    tf.keras.utils.set_random_seed(config.seed)
    train = training.reset_index(drop=True)
    valid = validation.reset_index(drop=True)
    model = build_direct_student(
        train_xyz_m=train.loc[:, XYZ_COLUMNS].to_numpy(float),
        zero_x_m=zero_x_m,
        beta_bounds_rad=beta_bounds_rad,
        config=config,
        include_signed_power=include_signed_power,
    )
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    coordinate = tf.constant(np.square(np.asarray(beta_coordinate_weights, dtype=np.float32)))
    rng = np.random.default_rng(config.seed)
    train_xyz = train.loc[:, XYZ_COLUMNS].to_numpy(np.float32)
    train_beta = train.loc[:, BETA_COLUMNS].to_numpy(np.float32)
    train_weight = (
        train.get("sample_weight", pd.Series(1.0, index=train.index))
        .fillna(1.0)
        .to_numpy(np.float32)
    )
    jacobian_columns = JACOBIAN_COLUMNS
    train_jacobian = (
        train.loc[:, jacobian_columns].to_numpy(np.float32).reshape(-1, 3, 6)
        if set(jacobian_columns) <= set(train)
        else None
    )
    valid_xyz = valid.loc[:, XYZ_COLUMNS].to_numpy(np.float32)
    valid_beta = valid.loc[:, BETA_COLUMNS].to_numpy(np.float32)
    id_to_index = {str(target): index for index, target in enumerate(train["target_id"].astype(str))}
    edge_pairs: list[tuple[int, int]] = []
    edge_weight: list[float] = []
    for row in train_edges.to_dict("records"):
        left, right = str(row["left_target_id"]), str(row["right_target_id"])
        if left in id_to_index and right in id_to_index:
            edge_pairs.append((id_to_index[left], id_to_index[right]))
            edge_weight.append(float(row.get("weight", row.get("w_ij", 1.0))))
    edges = np.asarray(edge_pairs, dtype=np.int64).reshape(-1, 2)
    edge_weight_array = np.asarray(edge_weight, dtype=np.float32)
    history: list[dict[str, Any]] = []
    best = math.inf
    best_weights: list[np.ndarray] | None = None
    stale = 0
    for step in range(1, config.maximum_steps + 1):
        batch = rng.choice(len(train), size=min(config.batch_size, len(train)), replace=False)
        with tf.GradientTape() as tape:
            prediction = model(train_xyz[batch], training=True)
            point = tf.reduce_sum(coordinate * tf.square(prediction - train_beta[batch]), axis=1)
            beta_loss = tf.reduce_sum(point * train_weight[batch]) / tf.reduce_sum(train_weight[batch])
            edge_loss = tf.constant(0.0, dtype=prediction.dtype)
            jacobian_loss = tf.constant(0.0, dtype=prediction.dtype)
            if config.edge_lambda > 0 and len(edges):
                edge_positions = rng.choice(len(edges), size=min(config.batch_size, len(edges)), replace=False)
                edge_batch = edges[edge_positions]
                left, right = edge_batch[:, 0], edge_batch[:, 1]
                edge_xyz = np.concatenate((train_xyz[left], train_xyz[right]), axis=0)
                edge_prediction = model(edge_xyz, training=True)
                left_prediction, right_prediction = tf.split(edge_prediction, 2, axis=0)
                teacher_delta = train_beta[left] - train_beta[right]
                prediction_delta = left_prediction - right_prediction
                distance_mm = np.linalg.norm(train_xyz[left] - train_xyz[right], axis=1) * 1000.0
                rate = (prediction_delta - teacher_delta) / tf.constant(
                    np.maximum(distance_mm, 5.0)[:, None], dtype=prediction.dtype
                )
                edge_point = tf.reduce_sum(coordinate * tf.square(rate), axis=1)
                selected_weights = edge_weight_array[edge_positions] if len(edge_weight_array) else np.ones(len(edge_point), np.float32)
                edge_loss = tf.reduce_sum(edge_point * selected_weights) / tf.reduce_sum(selected_weights)
            if config.jacobian_lambda > 0:
                if train_jacobian is None:
                    raise ValueError("Jacobian Direct Student requires registered jacobian_0_0..jacobian_2_5 columns")
                beta_error = prediction - train_beta[batch]
                task_error = tf.einsum("bij,bj->bi", train_jacobian[batch], beta_error) / tf.cast(0.003, prediction.dtype)
                jacobian_loss = tf.reduce_mean(tf.reduce_sum(tf.square(task_error), axis=1))
            loss = (
                beta_loss
                + tf.cast(config.edge_lambda, beta_loss.dtype) * edge_loss
                + tf.cast(config.jacobian_lambda, beta_loss.dtype) * jacobian_loss
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
        if step % config.validation_interval != 0 and step != config.maximum_steps:
            continue
        valid_prediction = model(valid_xyz, training=False)
        validation_loss = float(tf.reduce_mean(tf.reduce_sum(coordinate * tf.square(valid_prediction - valid_beta), axis=1)))
        history.append(
            {
                "step": step,
                "train_beta_loss": float(beta_loss),
                "train_edge_loss": float(edge_loss),
                "train_jacobian_loss": float(jacobian_loss),
                "validation_beta_loss": validation_loss,
            }
        )
        if validation_loss < best - 1.0e-12:
            best = validation_loss
            best_weights = model.get_weights()
            stale = 0
        else:
            stale += 1
            if stale >= config.patience_intervals:
                break
    if best_weights is not None:
        model.set_weights(best_weights)
    return model, pd.DataFrame(history)


def classify_causal_evidence(contrasts: Mapping[str, Any]) -> Mapping[str, Any]:
    """Apply the retry19 preregistered, non-exclusive cause rules."""

    supported: list[str] = []
    q0 = float(contrasts.get("q0_augmented_raw_max_spike_mm", math.inf))
    f0 = float(contrasts.get("f0_direct_raw_max_spike_mm", math.inf))
    if math.isfinite(q0) and q0 > 0 and f0 <= 0.5 * q0 and bool(contrasts.get("f0_nonregression", False)):
        supported.append("quotient_representation")
    best_without_j = float(contrasts.get("best_without_jacobian_raw_max_spike_mm", math.inf))
    with_j = float(contrasts.get("jacobian_raw_max_spike_mm", math.inf))
    if math.isfinite(best_without_j) and best_without_j > 0 and with_j <= 0.75 * best_without_j and bool(
        contrasts.get("jacobian_crosses_raw_gate", False)
    ):
        supported.append("jacobian_sensitivity")
    if bool(contrasts.get("expanded_data_only_crosses_raw_gate", False)) and bool(
        contrasts.get("local_support_distance_improved", False)
    ):
        supported.append("local_data_support")
    if bool(contrasts.get("teacher_static_section_failed", False)):
        supported.append("teacher_field_non_single_valued")
    if set(supported) >= {"quotient_representation", "local_data_support"}:
        primary = "mixed_wrapper_and_local_support"
    elif set(supported) >= {"quotient_representation", "jacobian_sensitivity"}:
        primary = "mixed_representation_and_jacobian"
    elif supported:
        primary = supported[0]
    else:
        primary = "inconclusive"
    seed_passes = int(contrasts.get("replicated_seed_pass_count", 0))
    if seed_passes >= 3 and supported:
        strength = "strong"
    elif seed_passes >= 1 and supported:
        strength = "moderate"
    elif supported:
        strength = "weak"
    else:
        strength = "inconclusive"
    return {
        "primary_cause_class": primary,
        "secondary_supported_factors": [value for value in supported if value != primary],
        "causal_evidence_strength": strength,
        "supported_factors": supported,
    }


__all__ = [
    "DirectStudentConfig",
    "aligned_q0_augmented_and_f0_rows",
    "build_direct_student",
    "classify_causal_evidence",
    "direct_feature_matrix",
    "edge_delta_loss_numpy",
    "load_direct_student",
    "signed_macroblock_ids",
    "smooth_signed_power",
    "smooth_signed_power_derivative",
    "train_direct_student",
    "unified_split_registry",
]
