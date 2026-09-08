"""Retention-aware single-Student distillation for BACRA V12.15.

The module keeps two policy seams deliberately small:

* whole spatial macro blocks are assigned before training, and
* a single bounded Student is trained against both the canonical Teacher
  labels and the already-validated V12.14 composite predictions.

It does not load or evaluate the sealed holdout.  The runner owns that later
stage after the selected model hashes have been locked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .bacra_exploratory_expansion import point_margin_deg
from .dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from .region_growth import JACOBIAN_COLUMNS, pack_voxels, voxel_indices
from .student_tracking_tf import StudentGeometry


DISTILL_BETA_COLUMNS = tuple(f"distill_{name}" for name in BETA_COLUMNS)


@dataclass(frozen=True)
class SpatialBlockPolicy:
    """Frozen whole-block partition policy."""

    macro_voxel_mm: float = 15.0
    validation_fraction: float = 0.15
    sealed_holdout_fraction: float = 0.15
    train_buffer_mm: float = 5.0

    def __post_init__(self) -> None:
        if not np.isfinite(
            [
                self.macro_voxel_mm,
                self.validation_fraction,
                self.sealed_holdout_fraction,
                self.train_buffer_mm,
            ]
        ).all():
            raise ValueError("spatial block policy values must be finite")
        if self.macro_voxel_mm <= 0.0 or self.train_buffer_mm < 0.0:
            raise ValueError("voxel size must be positive and buffer non-negative")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0,1)")
        if not 0.0 < self.sealed_holdout_fraction < 1.0:
            raise ValueError("sealed_holdout_fraction must lie in (0,1)")
        if self.validation_fraction + self.sealed_holdout_fraction >= 1.0:
            raise ValueError("validation and sealed fractions must leave train blocks")


def spatial_block_keys(xyz_m: np.ndarray, macro_voxel_mm: float) -> np.ndarray:
    """Return deterministic packed macro-block keys for task-space points."""

    xyz = np.asarray(xyz_m, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz_m must have finite shape (N,3)")
    return pack_voxels(voxel_indices(xyz, float(macro_voxel_mm)))


def assign_whole_spatial_blocks(
    frame: pd.DataFrame,
    *,
    region_mask: Sequence[bool],
    policy: SpatialBlockPolicy,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign whole region blocks to train, validation, or sealed holdout.

    Block selection is based only on task coordinates.  The same assignment is
    then applied to every row, including replay rows, so no old-corridor sample
    can leak into a sealed region block.
    """

    output = frame.copy()
    mask = np.asarray(region_mask, dtype=bool)
    if mask.shape != (len(output),) or not mask.any():
        raise ValueError("region_mask must select at least one row")
    keys = spatial_block_keys(
        output.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
        policy.macro_voxel_mm,
    )
    region_keys = np.unique(keys[mask])
    if len(region_keys) < 3:
        raise ValueError("at least three region macro blocks are required")
    scored = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{int(seed)}:{int(key)}".encode()).digest()[:8],
                "big",
            )
            for key in region_keys
        ],
        dtype=np.uint64,
    )
    order = np.argsort(scored, kind="stable")
    sealed_count = max(
        1, int(round(len(region_keys) * policy.sealed_holdout_fraction))
    )
    validation_count = max(
        1, int(round(len(region_keys) * policy.validation_fraction))
    )
    if sealed_count + validation_count >= len(region_keys):
        raise ValueError("registered fractions leave no training macro block")
    sealed_keys = np.sort(region_keys[order[:sealed_count]])
    validation_keys = np.sort(
        region_keys[order[sealed_count : sealed_count + validation_count]]
    )

    role = np.full(len(output), "train", dtype=object)
    role[np.isin(keys, validation_keys)] = "validation"
    role[np.isin(keys, sealed_keys)] = "sealed_holdout"
    if policy.train_buffer_mm > 0.0:
        withheld = role != "train"
        if withheld.any() and (~withheld).any():
            distance, _ = cKDTree(
                output.loc[withheld, XYZ_COLUMNS].to_numpy(dtype=float)
            ).query(
                output.loc[~withheld, XYZ_COLUMNS].to_numpy(dtype=float),
                k=1,
            )
            train_index = np.flatnonzero(~withheld)
            buffered = train_index[
                np.asarray(distance, dtype=float) * 1000.0
                < float(policy.train_buffer_mm)
            ]
            role[buffered] = "buffer_excluded"

    output["spatial_block_key"] = keys
    output["v12_15_split"] = role
    report = {
        **asdict(policy),
        "partition_seed": int(seed),
        "source_rows": int(len(output)),
        "region_macro_block_count": int(len(region_keys)),
        "train_macro_block_count": int(
            len(set(map(int, region_keys)) - set(map(int, sealed_keys)) - set(map(int, validation_keys)))
        ),
        "validation_macro_block_count": int(len(validation_keys)),
        "sealed_macro_block_count": int(len(sealed_keys)),
        "train_rows": int(np.count_nonzero(role == "train")),
        "validation_rows": int(np.count_nonzero(role == "validation")),
        "sealed_rows": int(np.count_nonzero(role == "sealed_holdout")),
        "buffer_excluded_rows": int(np.count_nonzero(role == "buffer_excluded")),
        "validation_block_keys": [int(value) for value in validation_keys],
        "sealed_block_keys": [int(value) for value in sealed_keys],
    }
    return output, report


def _frame_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        *XYZ_COLUMNS,
        *BETA_COLUMNS,
        *DISTILL_BETA_COLUMNS,
        *JACOBIAN_COLUMNS,
        "sample_weight",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"distillation frame missing columns: {missing}")
    return (
        frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32),
        frame.loc[:, BETA_COLUMNS].to_numpy(dtype=np.float32),
        frame.loc[:, DISTILL_BETA_COLUMNS].to_numpy(dtype=np.float32),
        frame["sample_weight"].to_numpy(dtype=np.float32),
        frame.loc[:, JACOBIAN_COLUMNS]
        .to_numpy(dtype=np.float32)
        .reshape(-1, 3, 6),
    )


def _validate_sampling_weights(
    frame: pd.DataFrame, sampling_weights: Mapping[str, float]
) -> tuple[list[str], list[float]]:
    observed = sorted(map(str, frame["sampling_bucket"].unique()))
    registered = {str(key): float(value) for key, value in sampling_weights.items()}
    if set(observed) != set(registered):
        raise ValueError(
            "sampling weights must exactly cover observed buckets: "
            f"observed={observed}, registered={sorted(registered)}"
        )
    values = np.asarray([registered[name] for name in observed], dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or not np.isclose(values.sum(), 1.0)
    ):
        raise ValueError("sampling weights must be positive and sum to one")
    return observed, values.tolist()


def train_retention_distilled_student(
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
    sampling_weights: Mapping[str, float],
    teacher_beta_weight: float,
    distill_beta_weight: float,
    fk_weight: float,
    margin_weight: float,
    row_loss_weight: float,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Fine-tune one bounded MLP against Teacher and composite targets."""

    import tensorflow as tf
    from quasi_exp.model.kinematics_tf import forward_xyz_from_beta_tf

    for value in (
        learning_rate,
        teacher_beta_weight,
        distill_beta_weight,
        fk_weight,
        margin_weight,
        row_loss_weight,
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("loss weights and learning rate must be finite/non-negative")
    if learning_rate <= 0.0 or teacher_beta_weight + distill_beta_weight <= 0.0:
        raise ValueError("positive learning rate and beta supervision are required")
    if not len(training_frame) or not len(validation_frame):
        raise ValueError("training and validation frames must be non-empty")
    if "sampling_bucket" not in training_frame:
        raise ValueError("training frame requires sampling_bucket")

    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    train_arrays = _frame_arrays(training_frame)
    valid_arrays = _frame_arrays(validation_frame)
    buckets, bucket_weights = _validate_sampling_weights(
        training_frame, sampling_weights
    )
    sources = []
    for offset, bucket in enumerate(buckets):
        selected = training_frame["sampling_bucket"].eq(bucket).to_numpy()
        arrays = tuple(value[selected] for value in train_arrays)
        sources.append(
            tf.data.Dataset.from_tensor_slices(arrays)
            .shuffle(
                int(np.count_nonzero(selected)),
                seed=int(seed) + offset,
                reshuffle_each_iteration=True,
            )
            .repeat()
        )
    dataset = tf.data.Dataset.sample_from_datasets(
        sources,
        weights=bucket_weights,
        seed=int(seed),
        stop_on_empty_dataset=False,
    )
    iterator = iter(
        dataset.batch(int(batch_size), drop_remainder=False).prefetch(
            tf.data.AUTOTUNE
        )
    )

    bounds = tf.constant(
        np.asarray(geometry.beta_bounds_rad, dtype=np.float32), dtype=tf.float32
    )
    beta_scale = tf.constant(
        np.deg2rad(np.asarray([1, 1, 1, 1, 0.75, 1], dtype=np.float32))
    )
    optimizer = tf.keras.optimizers.Adam(float(learning_rate))

    def loss_terms(batch: tuple[Any, ...], training: bool) -> tuple[Any, ...]:
        xyz, beta_teacher, beta_distill, sample_weight, jac = batch
        prediction = model(xyz, training=training)
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
                (predicted_xyz - xyz) / tf.cast(0.003, prediction.dtype)
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
            tf.square(projected / tf.cast(0.003, prediction.dtype)), axis=1
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
            tf.reduce_mean(row_point),
        )

    best = math.inf
    best_weights = model.get_weights()
    stale = 0
    history_rows: list[dict[str, Any]] = []
    valid_tensor = tuple(tf.convert_to_tensor(value) for value in valid_arrays)
    for step in range(1, int(max_steps) + 1):
        batch = next(iterator)
        with tf.GradientTape() as tape:
            objective, *_ = loss_terms(batch, True)
        gradients = tape.gradient(objective, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if step % int(validation_interval) != 0 and step != int(max_steps):
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
                "validation_row_loss": float(terms[4].numpy()),
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
    history = pd.DataFrame(history_rows)
    validation_prediction = np.asarray(
        model.predict(
            validation_frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32),
            batch_size=2048,
            verbose=0,
        ),
        dtype=float,
    )
    teacher_beta = validation_frame.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    distill_beta = validation_frame.loc[:, DISTILL_BETA_COLUMNS].to_numpy(
        dtype=float
    )
    return model, history, {
        "seed": int(seed),
        "training_rows": int(len(training_frame)),
        "validation_rows": int(len(validation_frame)),
        "optimizer_steps": int(history["step"].max()) if len(history) else 0,
        "best_validation_objective": float(best),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "sampling_weights": {
            name: float(value)
            for name, value in zip(buckets, bucket_weights, strict=True)
        },
        "loss_weights": {
            "teacher_beta": float(teacher_beta_weight),
            "distill_beta": float(distill_beta_weight),
            "fk": float(fk_weight),
            "margin": float(margin_weight),
            "row": float(row_loss_weight),
        },
        "validation_teacher_beta_rms_p95_deg": float(
            np.percentile(
                np.rad2deg(
                    np.sqrt(
                        np.mean(
                            np.square(validation_prediction - teacher_beta),
                            axis=1,
                        )
                    )
                ),
                95,
            )
        ),
        "validation_distill_beta_rms_p95_deg": float(
            np.percentile(
                np.rad2deg(
                    np.sqrt(
                        np.mean(
                            np.square(validation_prediction - distill_beta),
                            axis=1,
                        )
                    )
                ),
                95,
            )
        ),
        "validation_minimum_margin_deg": float(
            np.min(
                point_margin_deg(
                    validation_prediction,
                    np.asarray(geometry.beta_bounds_rad, dtype=float),
                )
            )
        ),
    }
