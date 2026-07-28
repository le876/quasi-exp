"""Student selection and one-seed training for BACRA-V12."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from .student_tracking_tf import (
    StudentGeometry,
    build_static_model,
    compile_student,
)


class StudentStrategy(str, Enum):
    STATIC = "static_xyz_to_beta6"
    CHART_CONDITIONED = "chart_conditioned_xyz_chart_to_beta6"
    STATEFUL = "stateful_tracking_policy"


@dataclass(frozen=True)
class DeviceExecutionPlan:
    device: str
    requested_workers: int
    effective_workers: int
    seed_order: tuple[int, ...]
    intraop_threads: int
    interop_threads: int


def choose_student_strategy(audit_report: Mapping[str, Any]) -> StudentStrategy:
    """Map atlas evidence to the only scientifically authorized Student class."""

    if bool(audit_report.get("static_inverse_authorized", False)):
        return StudentStrategy.STATIC
    if bool(audit_report.get("chart_conditioned_inverse_authorized", False)):
        return StudentStrategy.CHART_CONDITIONED
    return StudentStrategy.STATEFUL


def plan_seed_execution(
    seeds: Sequence[int],
    *,
    gpu_available: bool,
    cpu_workers: int,
    intraop_threads: int,
    interop_threads: int,
) -> DeviceExecutionPlan:
    ordered = tuple(int(value) for value in seeds)
    if not ordered:
        raise ValueError("at least one Student seed is required")
    requested = 1 if gpu_available else int(cpu_workers)
    effective = 1 if gpu_available else min(len(ordered), max(1, requested))
    return DeviceExecutionPlan(
        device="gpu:0" if gpu_available else "cpu",
        requested_workers=requested,
        effective_workers=effective,
        seed_order=ordered,
        intraop_threads=int(intraop_threads),
        interop_threads=int(interop_threads),
    )


def _require_split(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    required = {*XYZ_COLUMNS, *BETA_COLUMNS, "split_role"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Student dataset missing columns: {missing}")
    selected = frame[frame["split_role"].eq(role)].copy()
    if len(selected) == 0:
        raise ValueError(f"Student dataset has no {role} rows")
    return selected


def _chart_model(
    train: pd.DataFrame,
    *,
    geometry: StudentGeometry,
    hidden_units: Sequence[int],
) -> tuple[Any, tuple[str, ...]]:
    import tensorflow as tf

    charts = tuple(sorted(train["chart_id"].astype(str).unique()))
    chart_index = {value: index for index, value in enumerate(charts)}
    xyz = train[list(XYZ_COLUMNS)].to_numpy(dtype=np.float32)
    normalization = tf.keras.layers.Normalization(name="xyz_normalization")
    normalization.adapt(xyz)
    inputs = tf.keras.Input((3 + len(charts),), name="xyz_chart_context")
    normalized_xyz = normalization(inputs[:, :3])
    hidden = tf.keras.layers.Concatenate()([normalized_xyz, inputs[:, 3:]])
    for index, units in enumerate(hidden_units):
        hidden = tf.keras.layers.Dense(
            int(units), activation="gelu", name=f"dense_{index}"
        )(hidden)
    latent = tf.keras.layers.Dense(6, name="beta_latent")(hidden)
    bounds = np.asarray(geometry.beta_bounds_rad, dtype=np.float32)
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    halfspan = 0.5 * (bounds[:, 1] - bounds[:, 0])
    beta = tf.keras.layers.Rescaling(
        halfspan, offset=midpoint, name="beta_rad"
    )(tf.keras.layers.Activation("tanh", name="beta_unit")(latent))
    return tf.keras.Model(inputs, beta, name="chart_conditioned_student"), charts


def _features(frame: pd.DataFrame, charts: Sequence[str] | None) -> np.ndarray:
    xyz = frame[list(XYZ_COLUMNS)].to_numpy(dtype=np.float32)
    if charts is None:
        return xyz
    lookup = {str(value): index for index, value in enumerate(charts)}
    one_hot = np.zeros((len(frame), len(charts)), dtype=np.float32)
    for row, value in enumerate(frame["chart_id"].astype(str)):
        if value not in lookup:
            raise ValueError(f"validation contains unseen chart_id {value}")
        one_hot[row, lookup[value]] = 1.0
    return np.column_stack([xyz, one_hot])


def _packed_targets(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame[list(BETA_COLUMNS)].to_numpy(dtype=np.float32),
            frame[list(XYZ_COLUMNS)].to_numpy(dtype=np.float32),
        ]
    )


def train_one_seed(
    frame: pd.DataFrame,
    *,
    strategy: StudentStrategy,
    geometry: StudentGeometry,
    seed: int,
    hidden_units: Sequence[int],
    lambda_fk: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    output_dir: str | Path,
    beta_loss_scale_deg: float | Sequence[float] | None = None,
    beta5_head_units: Sequence[int] = (),
) -> Mapping[str, Any]:
    """Train one deterministic bounded TensorFlow Student.

    Stateful training is intentionally not synthesized from unordered regional
    points; callers must supply a trajectory-specific trainer instead.
    """

    if strategy is StudentStrategy.STATEFUL:
        raise RuntimeError(
            "stateful Student requires frozen sequential trajectory windows"
        )
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    train = _require_split(frame, "train")
    validation = _require_split(frame, "validation")
    if strategy is StudentStrategy.STATIC:
        model = build_static_model(
            train[list(XYZ_COLUMNS)].to_numpy(dtype=np.float32),
            geometry=geometry,
            output_mode="tanh",
            hidden_units=tuple(int(value) for value in hidden_units),
            beta5_head_units=tuple(
                int(value) for value in beta5_head_units
            ),
        )
        charts: tuple[str, ...] | None = None
    else:
        if "chart_id" not in train or "chart_id" not in validation:
            raise ValueError("chart-conditioned Student requires chart_id")
        model, charts = _chart_model(
            train, geometry=geometry, hidden_units=hidden_units
        )
    compile_student(
        model,
        geometry=geometry,
        lambda_fk=float(lambda_fk),
        learning_rate=float(learning_rate),
        beta_loss_scale_deg=beta_loss_scale_deg,
    )
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(patience),
        restore_best_weights=True,
    )
    history = model.fit(
        _features(train, charts),
        _packed_targets(train),
        validation_data=(_features(validation, charts), _packed_targets(validation)),
        epochs=int(max_epochs),
        batch_size=int(batch_size),
        shuffle=True,
        verbose=0,
        callbacks=[callback],
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.keras")
    prediction = np.asarray(
        model.predict(_features(validation, charts), batch_size=int(batch_size), verbose=0),
        dtype=float,
    )
    truth = validation[list(BETA_COLUMNS)].to_numpy(dtype=float)
    beta_gap_deg = np.rad2deg(
        np.sqrt(np.mean(np.square(prediction - truth), axis=1))
    )
    beta_abs_gap_deg = np.abs(np.rad2deg(prediction - truth))
    from quasi_exp.model.kinematics_tf import forward_xyz_from_beta_tf

    predicted_xyz = np.asarray(
        forward_xyz_from_beta_tf(
            prediction.astype(np.float32),
            lengths_m=geometry.lengths_m,
            p_end_local_m=geometry.p_end_local_m,
            theta_sign=geometry.theta_sign,
        ),
        dtype=float,
    )
    target_xyz = validation[list(XYZ_COLUMNS)].to_numpy(dtype=float)
    fk_error_mm = np.linalg.norm(predicted_xyz - target_xyz, axis=1) * 1000.0
    report = {
        "seed": int(seed),
        "strategy": strategy.value,
        "lambda_fk": float(lambda_fk),
        "beta_loss_scale_deg": (
            None
            if beta_loss_scale_deg is None
            else np.broadcast_to(
                np.asarray(beta_loss_scale_deg, dtype=float), (6,)
            ).tolist()
        ),
        "beta5_head_units": list(map(int, beta5_head_units)),
        "chart_ids": list(charts or ()),
        "epoch_count": int(len(history.history["loss"])),
        "best_validation_loss": float(np.min(history.history["val_loss"])),
        "validation_beta_rms_p50_deg": float(np.percentile(beta_gap_deg, 50)),
        "validation_beta_rms_p95_deg": float(np.percentile(beta_gap_deg, 95)),
        "validation_beta_rms_max_deg": float(np.max(beta_gap_deg)),
        "validation_beta_abs_p95_by_joint_deg": np.percentile(
            beta_abs_gap_deg, 95, axis=0
        ).tolist(),
        "validation_beta_abs_max_by_joint_deg": np.max(
            beta_abs_gap_deg, axis=0
        ).tolist(),
        "validation_fk_p50_mm": float(np.percentile(fk_error_mm, 50)),
        "validation_fk_p95_mm": float(np.percentile(fk_error_mm, 95)),
        "validation_fk_max_mm": float(np.max(fk_error_mm)),
        "train_row_count": int(len(train)),
        "validation_row_count": int(len(validation)),
    }
    pd.DataFrame(history.history).to_csv(output / "history.csv", index=False)
    return report
