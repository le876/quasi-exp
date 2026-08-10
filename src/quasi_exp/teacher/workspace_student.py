"""Bounded, branch-aware Student models for the BACRA V14 workspace protocol.

This module deliberately has a small surface: it validates the materialized
supervision frame, trains the representation selected by the workspace atlas,
and exposes an :class:`WorkspaceInverse` for inference.  In particular,
``chart_id`` is a training label for router/expert supervision, never an
inference input.  Static labels are accepted as canonical only when the
dataset already marks them primary; no FK-residual comparison is used to
choose a canonical branch here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from .region_growth import JACOBIAN_COLUMNS
from .student_tracking_tf import StudentGeometry
from .workspace_atlas import RepresentationMode
from .workspace_dataset import SupervisionKind, SupervisionRecord
from .workspace_inverse import (
    InverseQuery,
    PredictionResult,
    WorkspaceInverse,
    WorkspaceInversePolicy,
    WorkspaceRegistry,
)


PREVIOUS_BETA_COLUMNS = tuple(f"previous_beta{index}_rad" for index in range(1, 7))
REQUIRED_STUDENT_COLUMNS = (
    "record_id",
    "kind",
    "split_role",
    "chart_id",
    "is_primary",
    "sample_weight",
    *XYZ_COLUMNS,
    *BETA_COLUMNS,
    *JACOBIAN_COLUMNS,
)


@dataclass(frozen=True)
class WorkspaceStudentLossWeights:
    """Positive weights for the three per-record supervised loss terms."""

    beta: float = 1.0
    fk: float = 1.0
    row_space: float = 1.0

    def __post_init__(self) -> None:
        values = (self.beta, self.fk, self.row_space)
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
            raise ValueError("Student beta, FK, and row-space loss weights must be finite and positive")


@dataclass(frozen=True)
class WorkspaceStudentTrainingConfig:
    hidden_units: tuple[int, ...] = (128, 128, 64)
    router_hidden_units: tuple[int, ...] = (128, 64)
    learning_rate: float = 3.0e-4
    max_steps: int = 1_000
    validation_interval: int = 50
    patience_intervals: int = 10
    seed: int = 20260805
    loss_weights: WorkspaceStudentLossWeights = field(
        default_factory=WorkspaceStudentLossWeights
    )

    def __post_init__(self) -> None:
        units = tuple(int(value) for value in self.hidden_units)
        router_units = tuple(int(value) for value in self.router_hidden_units)
        if not units or any(value <= 0 for value in units):
            raise ValueError("hidden_units must contain positive widths")
        if not router_units or any(value <= 0 for value in router_units):
            raise ValueError("router_hidden_units must contain positive widths")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if int(self.max_steps) < 1 or int(self.validation_interval) < 1:
            raise ValueError("max_steps and validation_interval must be positive")
        if int(self.patience_intervals) < 1:
            raise ValueError("patience_intervals must be positive")
        object.__setattr__(self, "hidden_units", units)
        object.__setattr__(self, "router_hidden_units", router_units)
        object.__setattr__(self, "max_steps", int(self.max_steps))
        object.__setattr__(self, "validation_interval", int(self.validation_interval))
        object.__setattr__(self, "patience_intervals", int(self.patience_intervals))
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class WorkspaceStudentModels:
    """Models for precisely one representation mode.

    ``chart_ids`` fixes router-column ordering and is copied into the inverse
    facade.  It must match the expert mapping exactly.
    """

    mode: RepresentationMode
    chart_ids: tuple[str, ...] = ()
    global_model: Any | None = None
    router_model: Any | None = None
    expert_models: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = RepresentationMode(self.mode)
        charts = tuple(str(value) for value in self.chart_ids)
        experts = dict(self.expert_models)
        if mode is RepresentationMode.XYZ_GLOBAL:
            if self.global_model is None or self.router_model is not None or experts or charts:
                raise ValueError("xyz_global requires exactly one global model")
        elif mode in (
            RepresentationMode.XYZ_ROUTER_EXPERTS,
            RepresentationMode.STATEFUL_ROUTER_EXPERTS,
        ):
            if self.global_model is not None or self.router_model is None:
                raise ValueError("router modes require router_model and no global model")
            if not charts or set(charts) != set(experts) or len(set(charts)) != len(charts):
                raise ValueError("router chart_ids must be unique and exactly match experts")
        else:
            raise ValueError("blocked representation cannot own Student models")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "chart_ids", charts)
        object.__setattr__(self, "expert_models", experts)


@dataclass(frozen=True)
class WorkspaceStudentTrainingResult:
    models: WorkspaceStudentModels
    inverse: WorkspaceInverse
    history: pd.DataFrame
    train_row_count: int
    validation_row_count: int


@dataclass(frozen=True)
class WorkspaceStudentEvaluation:
    prediction: PredictionResult
    accepted_count: int
    abstained_count: int
    accepted_fraction: float


def student_feature_columns(mode: RepresentationMode) -> tuple[str, ...]:
    """Return the complete inference feature schema for one representation."""

    representation = RepresentationMode(mode)
    if representation is RepresentationMode.XYZ_GLOBAL:
        return XYZ_COLUMNS
    if representation is RepresentationMode.XYZ_ROUTER_EXPERTS:
        return XYZ_COLUMNS
    if representation is RepresentationMode.STATEFUL_ROUTER_EXPERTS:
        return (*XYZ_COLUMNS, *PREVIOUS_BETA_COLUMNS)
    raise ValueError("blocked representation has no Student feature schema")


def validate_workspace_student_frame(
    frame: pd.DataFrame, *, mode: RepresentationMode
) -> pd.DataFrame:
    """Validate one unexpanded supervision frame and return a defensive copy."""

    representation = RepresentationMode(mode)
    if representation is RepresentationMode.BLOCKED:
        raise ValueError("blocked representation has no training frame")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("workspace Student supervision must be a pandas DataFrame")
    required = set(REQUIRED_STUDENT_COLUMNS)
    if representation is RepresentationMode.STATEFUL_ROUTER_EXPERTS:
        required.update(PREVIOUS_BETA_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"workspace Student frame missing columns: {missing}")
    if len(frame) == 0:
        raise ValueError("workspace Student frame must contain at least one row")
    result = frame.copy()
    if result["record_id"].isna().any() or result["record_id"].astype(str).str.strip().eq("").any():
        raise ValueError("workspace Student record_id values must be non-empty")
    if result["record_id"].astype(str).duplicated().any():
        raise ValueError("workspace Student frame must not expand or duplicate record_id rows")
    if result["chart_id"].isna().any() or result["chart_id"].astype(str).str.strip().eq("").any():
        raise ValueError("workspace Student chart_id values must be non-empty")
    numeric = [*XYZ_COLUMNS, *BETA_COLUMNS, *JACOBIAN_COLUMNS, "sample_weight"]
    if representation is RepresentationMode.STATEFUL_ROUTER_EXPERTS:
        numeric.extend(PREVIOUS_BETA_COLUMNS)
    values = result.loc[:, numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("workspace Student numeric supervision must be finite")
    if np.any(result["sample_weight"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("workspace Student sample_weight must be positive")
    kinds = result["kind"].astype(str).to_numpy()
    expected_kind = (
        SupervisionKind.STATEFUL.value
        if representation is RepresentationMode.STATEFUL_ROUTER_EXPERTS
        else SupervisionKind.STATIC.value
    )
    if not np.all(kinds == expected_kind):
        raise ValueError(f"{representation.value} requires only {expected_kind} supervision")
    if representation is RepresentationMode.XYZ_GLOBAL and not result["is_primary"].astype(bool).all():
        raise ValueError("xyz_global requires primary canonical supervision only")
    return result


def records_to_workspace_student_frame(
    records: Sequence[SupervisionRecord],
    *,
    jacobian_at_beta: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """Materialize one row per selected supervision record, without replication."""

    rows: list[dict[str, Any]] = []
    for record in records:
        jacobian = np.asarray(jacobian_at_beta(np.asarray(record.beta_rad, dtype=float)), dtype=float)
        if jacobian.shape != (3, 6) or not np.isfinite(jacobian).all():
            raise ValueError("jacobian_at_beta must return finite shape (3, 6)")
        row: dict[str, Any] = {
            "record_id": record.record_id,
            "kind": record.kind.value,
            "split_role": "" if record.split_role is None else record.split_role.value,
            "chart_id": record.chart_id,
            "is_primary": record.is_primary,
            "sample_weight": record.sample_weight,
        }
        row.update(dict(zip(XYZ_COLUMNS, record.xyz_m, strict=True)))
        row.update(dict(zip(BETA_COLUMNS, record.beta_rad, strict=True)))
        row.update(dict(zip(JACOBIAN_COLUMNS, jacobian.reshape(-1), strict=True)))
        if record.previous_beta_rad is not None:
            row.update(
                dict(zip(PREVIOUS_BETA_COLUMNS, record.previous_beta_rad, strict=True))
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _set_seed(seed: int) -> None:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _build_bounded_mlp(
    features: np.ndarray,
    *,
    geometry: StudentGeometry,
    input_width: int,
    hidden_units: Sequence[int],
    name: str,
) -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input((int(input_width),), name="workspace_features")
    normalization = tf.keras.layers.Normalization(name="feature_normalization")
    normalization.adapt(np.asarray(features, dtype=np.float32))
    hidden: Any = normalization(inputs)
    for index, units in enumerate(hidden_units):
        hidden = tf.keras.layers.Dense(int(units), activation="gelu", name=f"dense_{index}")(hidden)
    latent = tf.keras.layers.Dense(6, name="beta_latent")(hidden)
    bounds = np.asarray(geometry.beta_bounds_rad, dtype=np.float32)
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    halfspan = 0.5 * (bounds[:, 1] - bounds[:, 0])
    output = tf.keras.layers.Rescaling(
        halfspan, offset=midpoint, name="beta_rad"
    )(tf.keras.layers.Activation("tanh", name="beta_unit")(latent))
    return tf.keras.Model(inputs, output, name=name)


def _build_router(
    features: np.ndarray,
    *,
    input_width: int,
    chart_count: int,
    hidden_units: Sequence[int],
    name: str,
) -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input((int(input_width),), name="workspace_features")
    normalization = tf.keras.layers.Normalization(name="feature_normalization")
    normalization.adapt(np.asarray(features, dtype=np.float32))
    hidden: Any = normalization(inputs)
    for index, units in enumerate(hidden_units):
        hidden = tf.keras.layers.Dense(int(units), activation="gelu", name=f"dense_{index}")(hidden)
    output = tf.keras.layers.Dense(int(chart_count), activation="softmax", name="chart_probabilities")(hidden)
    return tf.keras.Model(inputs, output, name=name)


def build_workspace_student_models(
    train_frame: pd.DataFrame,
    *,
    mode: RepresentationMode,
    geometry: StudentGeometry,
    hidden_units: Sequence[int] = (128, 128, 64),
    router_hidden_units: Sequence[int] | None = None,
) -> WorkspaceStudentModels:
    """Build bounded models from exact training rows; this function does not fit."""

    representation = RepresentationMode(mode)
    train = validate_workspace_student_frame(train_frame, mode=representation)
    columns = list(student_feature_columns(representation))
    features = train.loc[:, columns].to_numpy(dtype=np.float32)
    if representation is RepresentationMode.XYZ_GLOBAL:
        return WorkspaceStudentModels(
            mode=representation,
            global_model=_build_bounded_mlp(
                features,
                geometry=geometry,
                input_width=len(columns),
                hidden_units=hidden_units,
                name="workspace_xyz_global_student",
            ),
        )
    charts = tuple(sorted(train["chart_id"].astype(str).unique()))
    router_units = hidden_units if router_hidden_units is None else router_hidden_units
    experts = {
        chart: _build_bounded_mlp(
            train.loc[train["chart_id"].astype(str).eq(chart), columns].to_numpy(dtype=np.float32),
            geometry=geometry,
            input_width=len(columns),
            hidden_units=hidden_units,
            name=f"workspace_expert_{index}",
        )
        for index, chart in enumerate(charts)
    }
    return WorkspaceStudentModels(
        mode=representation,
        chart_ids=charts,
        router_model=_build_router(
            features,
            input_width=len(columns),
            chart_count=len(charts),
            hidden_units=router_units,
            name="workspace_chart_router",
        ),
        expert_models=experts,
    )


def workspace_student_loss_terms(
    *,
    features: Any,
    beta_true: Any,
    xyz_true: Any,
    jacobian_true: Any,
    sample_weight: Any,
    model: Any,
    geometry: StudentGeometry,
    loss_weights: WorkspaceStudentLossWeights,
    training: bool,
) -> Mapping[str, Any]:
    """Weighted beta, exact-FK, and frozen-J row-space terms for one batch."""

    import tensorflow as tf
    from quasi_exp.model.kinematics_tf import forward_xyz_from_beta_tf

    prediction = model(features, training=training)
    beta_true = tf.convert_to_tensor(beta_true, dtype=prediction.dtype)
    xyz_true = tf.convert_to_tensor(xyz_true, dtype=prediction.dtype)
    jacobian = tf.convert_to_tensor(jacobian_true, dtype=prediction.dtype)
    weights = tf.convert_to_tensor(sample_weight, dtype=prediction.dtype)
    if weights.shape.rank != 1:
        weights = tf.reshape(weights, (-1,))
    bounds = tf.constant(geometry.beta_bounds_rad, dtype=prediction.dtype)
    beta_scale = tf.maximum(bounds[:, 1] - bounds[:, 0], tf.cast(1.0e-6, prediction.dtype))
    beta_point = tf.reduce_mean(tf.square((prediction - beta_true) / beta_scale), axis=1)
    predicted_xyz = forward_xyz_from_beta_tf(
        prediction,
        lengths_m=geometry.lengths_m,
        p_end_local_m=geometry.p_end_local_m,
        theta_sign=geometry.theta_sign,
    )
    fk_point = tf.reduce_mean(
        tf.square((predicted_xyz - xyz_true) / tf.cast(0.003, prediction.dtype)), axis=1
    )
    projected = tf.einsum("nij,nj->ni", jacobian, prediction - beta_true)
    row_point = tf.reduce_mean(
        tf.square(projected / tf.cast(0.003, prediction.dtype)), axis=1
    )
    point = (
        tf.cast(loss_weights.beta, prediction.dtype) * beta_point
        + tf.cast(loss_weights.fk, prediction.dtype) * fk_point
        + tf.cast(loss_weights.row_space, prediction.dtype) * row_point
    )
    denominator = tf.reduce_sum(weights)
    objective = tf.reduce_sum(point * weights) / denominator
    return {
        "objective": objective,
        "beta": tf.reduce_sum(beta_point * weights) / denominator,
        "fk": tf.reduce_sum(fk_point * weights) / denominator,
        "row_space": tf.reduce_sum(row_point * weights) / denominator,
        "prediction": prediction,
    }


def _arrays(frame: pd.DataFrame, *, mode: RepresentationMode) -> tuple[np.ndarray, ...]:
    columns = list(student_feature_columns(mode))
    return (
        frame.loc[:, columns].to_numpy(dtype=np.float32),
        frame.loc[:, BETA_COLUMNS].to_numpy(dtype=np.float32),
        frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32),
        frame.loc[:, JACOBIAN_COLUMNS].to_numpy(dtype=np.float32).reshape(-1, 3, 6),
        frame["sample_weight"].to_numpy(dtype=np.float32),
    )


def _fit_model(
    model: Any,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    mode: RepresentationMode,
    geometry: StudentGeometry,
    config: WorkspaceStudentTrainingConfig,
    seed_offset: int,
) -> list[dict[str, Any]]:
    import tensorflow as tf

    train_arrays = tuple(tf.convert_to_tensor(value) for value in _arrays(train, mode=mode))
    validation_arrays = tuple(tf.convert_to_tensor(value) for value in _arrays(validation, mode=mode))
    optimizer = tf.keras.optimizers.Adam(float(config.learning_rate))
    best = math.inf
    best_weights = model.get_weights()
    stale = 0
    history: list[dict[str, Any]] = []
    del seed_offset  # Seed is set once before all models; no row re-sampling occurs.
    for step in range(1, config.max_steps + 1):
        with tf.GradientTape() as tape:
            terms = workspace_student_loss_terms(
                features=train_arrays[0], beta_true=train_arrays[1], xyz_true=train_arrays[2],
                jacobian_true=train_arrays[3], sample_weight=train_arrays[4], model=model,
                geometry=geometry, loss_weights=config.loss_weights, training=True,
            )
        gradients = tape.gradient(terms["objective"], model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if step % config.validation_interval != 0 and step != config.max_steps:
            continue
        valid = workspace_student_loss_terms(
            features=validation_arrays[0], beta_true=validation_arrays[1], xyz_true=validation_arrays[2],
            jacobian_true=validation_arrays[3], sample_weight=validation_arrays[4], model=model,
            geometry=geometry, loss_weights=config.loss_weights, training=False,
        )
        value = float(valid["objective"].numpy())
        history.append({
            "step": step,
            "validation_objective": value,
            "validation_beta_loss": float(valid["beta"].numpy()),
            "validation_fk_loss": float(valid["fk"].numpy()),
            "validation_row_space_loss": float(valid["row_space"].numpy()),
        })
        if value < best - 1.0e-7:
            best, best_weights, stale = value, model.get_weights(), 0
        else:
            stale += 1
        if stale >= config.patience_intervals:
            break
    model.set_weights(best_weights)
    return history


def _predictor(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    def predict(features: np.ndarray) -> np.ndarray:
        output = model(np.asarray(features, dtype=np.float32), training=False)
        return np.asarray(output, dtype=float)

    return predict


def _geometry_fk(geometry: StudentGeometry) -> Callable[[np.ndarray], np.ndarray]:
    def fk(beta_rad: np.ndarray) -> np.ndarray:
        from quasi_exp.model.kinematics_tf import forward_xyz_from_beta_tf

        output = forward_xyz_from_beta_tf(
            np.asarray(beta_rad, dtype=np.float32), lengths_m=geometry.lengths_m,
            p_end_local_m=geometry.p_end_local_m, theta_sign=geometry.theta_sign,
        )
        return np.asarray(output, dtype=float)

    return fk


def build_workspace_student_inverse(
    models: WorkspaceStudentModels,
    *,
    geometry: StudentGeometry,
    registry: WorkspaceRegistry | None = None,
    policy: WorkspaceInversePolicy | None = None,
) -> WorkspaceInverse:
    """Expose models through the fail-closed, no-known-chart inverse facade."""

    if models.mode is RepresentationMode.XYZ_GLOBAL:
        return WorkspaceInverse(
            mode=models.mode, beta_bounds_rad=geometry.beta_bounds_rad, fk=_geometry_fk(geometry),
            global_predictor=_predictor(models.global_model), registry=registry, policy=policy,
        )
    return WorkspaceInverse(
        mode=models.mode,
        beta_bounds_rad=geometry.beta_bounds_rad,
        fk=_geometry_fk(geometry),
        router_predictor=_predictor(models.router_model),
        expert_predictors={chart: _predictor(models.expert_models[chart]) for chart in models.chart_ids},
        registry=registry,
        policy=policy,
    )


def train_workspace_student(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    mode: RepresentationMode,
    geometry: StudentGeometry,
    config: WorkspaceStudentTrainingConfig | None = None,
    registry: WorkspaceRegistry | None = None,
    inverse_policy: WorkspaceInversePolicy | None = None,
) -> WorkspaceStudentTrainingResult:
    """Train one selected V14 representation without row duplication or padding."""

    representation = RepresentationMode(mode)
    options = WorkspaceStudentTrainingConfig() if config is None else config
    train = validate_workspace_student_frame(train_frame, mode=representation)
    validation = validate_workspace_student_frame(validation_frame, mode=representation)
    _set_seed(options.seed)
    models = build_workspace_student_models(
        train,
        mode=representation,
        geometry=geometry,
        hidden_units=options.hidden_units,
        router_hidden_units=options.router_hidden_units,
    )
    histories: list[pd.DataFrame] = []
    if representation is RepresentationMode.XYZ_GLOBAL:
        history = _fit_model(
            models.global_model, train, validation, mode=representation, geometry=geometry,
            config=options, seed_offset=0,
        )
        histories.append(pd.DataFrame(history).assign(model_id="global"))
    else:
        chart_index = {chart: index for index, chart in enumerate(models.chart_ids)}
        router_targets = np.asarray(
            [chart_index[str(value)] for value in train["chart_id"]], dtype=np.int32
        )
        import tensorflow as tf

        # Router fitting uses each original training row once per optimizer step;
        # this is classification-only and intentionally has no artificial samples.
        router_features = train.loc[:, student_feature_columns(representation)].to_numpy(dtype=np.float32)
        router_weights = train["sample_weight"].to_numpy(dtype=np.float32)
        router_validation = validation.loc[:, student_feature_columns(representation)].to_numpy(dtype=np.float32)
        valid_targets = np.asarray(
            [chart_index.get(str(value), -1) for value in validation["chart_id"]], dtype=np.int32
        )
        if np.any(valid_targets < 0):
            raise ValueError("validation contains a chart unseen in router training")
        router_optimizer = tf.keras.optimizers.Adam(float(options.learning_rate))
        router_history: list[dict[str, Any]] = []
        best, best_weights, stale = math.inf, models.router_model.get_weights(), 0
        for step in range(1, options.max_steps + 1):
            with tf.GradientTape() as tape:
                probability = models.router_model(router_features, training=True)
                point = tf.keras.losses.sparse_categorical_crossentropy(router_targets, probability)
                objective = tf.reduce_sum(point * router_weights) / tf.reduce_sum(router_weights)
            gradients = tape.gradient(objective, models.router_model.trainable_variables)
            router_optimizer.apply_gradients(zip(gradients, models.router_model.trainable_variables))
            if step % options.validation_interval != 0 and step != options.max_steps:
                continue
            probability = models.router_model(router_validation, training=False)
            point = tf.keras.losses.sparse_categorical_crossentropy(valid_targets, probability)
            value = float(tf.reduce_mean(point).numpy())
            router_history.append({"step": step, "validation_objective": value})
            if value < best - 1.0e-7:
                best, best_weights, stale = value, models.router_model.get_weights(), 0
            else:
                stale += 1
            if stale >= options.patience_intervals:
                break
        models.router_model.set_weights(best_weights)
        histories.append(pd.DataFrame(router_history).assign(model_id="router"))
        for offset, chart in enumerate(models.chart_ids, start=1):
            expert_train = train.loc[train["chart_id"].astype(str).eq(chart)].copy()
            expert_validation = validation.loc[validation["chart_id"].astype(str).eq(chart)].copy()
            if len(expert_validation) == 0:
                raise ValueError(f"validation has no rows for expert chart {chart}")
            history = _fit_model(
                models.expert_models[chart], expert_train, expert_validation,
                mode=representation, geometry=geometry, config=options, seed_offset=offset,
            )
            histories.append(pd.DataFrame(history).assign(model_id=f"expert:{chart}"))
    history = pd.concat(histories, ignore_index=True, sort=False)
    inverse = build_workspace_student_inverse(
        models, geometry=geometry, registry=registry, policy=inverse_policy
    )
    return WorkspaceStudentTrainingResult(
        models=models, inverse=inverse, history=history,
        train_row_count=len(train), validation_row_count=len(validation),
    )


def evaluate_workspace_student(
    inverse: WorkspaceInverse, query: InverseQuery
) -> WorkspaceStudentEvaluation:
    """Run the public inference path; caller supplies no chart identifier."""

    prediction = inverse.predict(query)
    accepted_count = int(np.count_nonzero(prediction.accepted))
    total = len(prediction.accepted)
    return WorkspaceStudentEvaluation(
        prediction=prediction,
        accepted_count=accepted_count,
        abstained_count=total - accepted_count,
        accepted_fraction=float(accepted_count / total) if total else 0.0,
    )


def save_workspace_student_models(
    models: WorkspaceStudentModels, output_dir: str | Path
) -> Mapping[str, Any]:
    """Persist uncompiled Keras models plus explicit router-column identity."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, str] = {}
    if models.mode is RepresentationMode.XYZ_GLOBAL:
        path = root / "global.keras"
        models.global_model.save(path, include_optimizer=False)
        artifacts["global"] = path.name
    else:
        router_path = root / "router.keras"
        models.router_model.save(router_path, include_optimizer=False)
        artifacts["router"] = router_path.name
        for index, chart_id in enumerate(models.chart_ids):
            path = root / f"expert_{index:03d}.keras"
            models.expert_models[chart_id].save(path, include_optimizer=False)
            artifacts[f"expert:{chart_id}"] = path.name
    manifest = {
        "schema_version": 1,
        "representation_mode": models.mode.value,
        "chart_ids": list(models.chart_ids),
        "artifacts": artifacts,
    }
    (root / "model_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_workspace_student_models(output_dir: str | Path) -> WorkspaceStudentModels:
    """Load exactly the models named by a persisted V14 model manifest."""

    import tensorflow as tf

    root = Path(output_dir)
    manifest = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
    mode = RepresentationMode(str(manifest["representation_mode"]))
    charts = tuple(map(str, manifest.get("chart_ids", ())))
    artifacts = {str(key): str(value) for key, value in manifest["artifacts"].items()}
    if mode is RepresentationMode.XYZ_GLOBAL:
        return WorkspaceStudentModels(
            mode=mode,
            global_model=tf.keras.models.load_model(
                root / artifacts["global"], compile=False
            ),
        )
    experts = {
        chart: tf.keras.models.load_model(
            root / artifacts[f"expert:{chart}"], compile=False
        )
        for chart in charts
    }
    return WorkspaceStudentModels(
        mode=mode,
        chart_ids=charts,
        router_model=tf.keras.models.load_model(
            root / artifacts["router"], compile=False
        ),
        expert_models=experts,
    )
