"""BACRA-V12.13 exploratory geometry expansion and Student repair.

The module deliberately keeps the V12.11/V12.12 protocol implementations
unchanged.  It provides the smaller V12.13 policy seam: exploratory label
admission, deterministic bridge geometry, task-sensitive diagnostics and
few-step DLS correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import qmc

from .canonical import weighted_damped_pinv
from .dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from .region import EllipseFamilySpec
from .student_tracking_tf import StudentGeometry, build_static_model


class ExploratoryEnvironment(Protocol):
    bounds: np.ndarray

    def fk(self, beta_rad: np.ndarray) -> np.ndarray: ...

    def jacobian(self, beta_rad: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ExploratoryTeacherPolicy:
    """Frozen V12.13 simulation-training label policy."""

    residual_p95_mm: float = 1.0
    residual_max_mm: float = 3.0
    seam_rms_deg: float = 1.0
    training_transition_rms_max_deg: float = 3.0
    legacy_transition_rms_max_deg: float = 2.0
    gold_margin_deg: float = 1.5
    silver_margin_deg: float = 0.25

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("teacher policy values must be finite and non-negative")
        if self.gold_margin_deg <= self.silver_margin_deg:
            raise ValueError("gold margin must be above silver margin")
        if (
            self.training_transition_rms_max_deg
            < self.legacy_transition_rms_max_deg
        ):
            raise ValueError("training transition cannot be tighter than legacy")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def cyclic_update_rms_deg(beta_rad: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float)
    if beta.ndim != 2 or beta.shape[1] != 6 or len(beta) < 4:
        raise ValueError("beta_rad must have shape (N, 6), N >= 4")
    return np.rad2deg(
        np.sqrt(np.mean(np.square(np.roll(beta, -1, axis=0) - beta), axis=1))
    )


def point_margin_deg(beta_rad: np.ndarray, bounds_rad: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    bounds = np.asarray(bounds_rad, dtype=float).reshape(6, 2)
    return np.rad2deg(
        np.min(
            np.minimum(
                beta - bounds[:, 0][None, :],
                bounds[:, 1][None, :] - beta,
            ),
            axis=1,
        )
    )


def evaluate_exploratory_teacher_label(
    beta_rad: np.ndarray,
    target_xyz_m: np.ndarray,
    achieved_xyz_m: np.ndarray,
    bounds_rad: np.ndarray,
    *,
    policy: ExploratoryTeacherPolicy,
    chart_id: str,
    root_chart_connection: bool,
) -> dict[str, Any]:
    """Classify one complete loop without conflating root connection and labels."""

    beta = np.asarray(beta_rad, dtype=float)
    target = np.asarray(target_xyz_m, dtype=float)
    achieved = np.asarray(achieved_xyz_m, dtype=float)
    bounds = np.asarray(bounds_rad, dtype=float).reshape(6, 2)
    shape_ok = bool(
        beta.ndim == 2
        and beta.shape[1:] == (6,)
        and target.shape == achieved.shape == (len(beta), 3)
        and len(beta) >= 4
    )
    finite = bool(
        shape_ok
        and np.isfinite(beta).all()
        and np.isfinite(target).all()
        and np.isfinite(achieved).all()
    )
    if finite:
        residual_mm = np.linalg.norm(achieved - target, axis=1) * 1000.0
        transition = cyclic_update_rms_deg(beta)
        margin = point_margin_deg(beta, bounds)
        actual_bounds = bool(
            np.all(beta >= bounds[:, 0][None, :])
            and np.all(beta <= bounds[:, 1][None, :])
        )
        residual_p95 = float(np.percentile(residual_mm, 95))
        residual_max = float(np.max(residual_mm))
        transition_p95 = float(np.percentile(transition, 95))
        transition_max = float(np.max(transition))
        seam = float(transition[-1])
        margin_min = float(np.min(margin))
    else:
        residual_p95 = residual_max = math.inf
        transition_p95 = transition_max = seam = math.inf
        margin_min = -math.inf
        actual_bounds = False

    checks = {
        "shape_and_phase_complete": shape_ok,
        "all_fields_finite": finite,
        "actual_mechanical_bounds": actual_bounds,
        "fk_residual": bool(
            residual_p95 <= policy.residual_p95_mm
            and residual_max <= policy.residual_max_mm
        ),
        "cyclic_seam": bool(seam <= policy.seam_rms_deg),
        "training_transition": bool(
            transition_max <= policy.training_transition_rms_max_deg
        ),
    }
    label_valid = bool(all(checks.values()))
    if not label_valid or margin_min < policy.silver_margin_deg:
        quality = "Reject"
    elif margin_min >= policy.gold_margin_deg:
        quality = "Gold"
    else:
        quality = "Silver"
    return {
        "schema_version": 1,
        "gate_semantics": "simulation_exploratory_training_label",
        "policy_fingerprint": policy.fingerprint,
        "chart_id": str(chart_id),
        "root_chart_connection": bool(root_chart_connection),
        "root_chart_connection_required_for_label": False,
        "quality_class": quality,
        "label_valid": label_valid,
        "training_eligible": bool(label_valid and quality in {"Gold", "Silver"}),
        "legacy_2deg_transition_pass": bool(
            finite and transition_max <= policy.legacy_transition_rms_max_deg
        ),
        "checks": checks,
        "metrics": {
            "residual_p95_mm": residual_p95,
            "residual_max_mm": residual_max,
            "transition_rms_p95_deg": transition_p95,
            "transition_rms_max_deg": transition_max,
            "seam_rms_deg": seam,
            "minimum_joint_margin_deg": margin_min,
        },
    }


def _frame_matrix(row: Mapping[str, Any]) -> np.ndarray:
    major = np.asarray(
        [row["major_x"], row["major_y"], row["major_z"]], dtype=float
    )
    minor = np.asarray(
        [row["minor_x"], row["minor_y"], row["minor_z"]], dtype=float
    )
    major /= np.linalg.norm(major)
    minor -= major * float(np.dot(major, minor))
    minor /= np.linalg.norm(minor)
    normal = np.cross(major, minor)
    normal /= np.linalg.norm(normal)
    return np.column_stack([major, minor, normal])


def _catalog_row(
    *,
    family_id: str,
    group_id: str,
    order: int,
    center_m: np.ndarray,
    frame: np.ndarray,
    major_semiaxis_m: float,
    axis_ratio: float,
    source: str,
    interpolation_s: float | None,
    generation_seed: int,
) -> dict[str, Any]:
    family = EllipseFamilySpec(
        family_id=family_id,
        center_m=np.asarray(center_m, dtype=float),
        major_direction=np.asarray(frame[:, 0], dtype=float),
        minor_direction=np.asarray(frame[:, 1], dtype=float),
        major_semiaxis_m=float(major_semiaxis_m),
        minor_semiaxis_m=float(major_semiaxis_m * axis_ratio),
        metadata={"group_id": group_id, "catalog_order": int(order)},
    )
    return {
        "family_id": family_id,
        "group_id": group_id,
        "catalog_order": int(order),
        "generation_seed": int(generation_seed),
        "major_semiaxis_m": float(major_semiaxis_m),
        "axis_ratio": float(axis_ratio),
        "center_x_m": float(center_m[0]),
        "center_y_m": float(center_m[1]),
        "center_z_m": float(center_m[2]),
        "major_x": float(frame[0, 0]),
        "major_y": float(frame[1, 0]),
        "major_z": float(frame[2, 0]),
        "minor_x": float(frame[0, 1]),
        "minor_y": float(frame[1, 1]),
        "minor_z": float(frame[2, 1]),
        "source": str(source),
        "interpolation_s": (
            None if interpolation_s is None else float(interpolation_s)
        ),
        "family_fingerprint": family.fingerprint,
    }


def generate_bridge_catalog(
    core_row: Mapping[str, Any],
    stress_row: Mapping[str, Any],
    *,
    lateral_seed: int = 20260760,
    lhs_seed: int = 20260761,
) -> pd.DataFrame:
    """Return the frozen 12 direct + 12 lateral + 8 LHS bridge catalog."""

    core_center = np.asarray(
        [core_row["center_x_m"], core_row["center_y_m"], core_row["center_z_m"]],
        dtype=float,
    )
    stress_center = np.asarray(
        [
            stress_row["center_x_m"],
            stress_row["center_y_m"],
            stress_row["center_z_m"],
        ],
        dtype=float,
    )
    core_frame = _frame_matrix(core_row)
    stress_frame = _frame_matrix(stress_row)
    rotations = Rotation.from_matrix(
        np.stack([core_frame, stress_frame], axis=0)
    )
    slerp = Slerp([0.0, 1.0], rotations)
    rows: list[dict[str, Any]] = []
    direct_s = np.arange(1, 13, dtype=float) / 13.0
    for index, s in enumerate(direct_s):
        rows.append(
            _catalog_row(
                family_id=f"bridge_direct_{index:02d}",
                group_id="bridge_direct",
                order=len(rows),
                center_m=(1.0 - s) * core_center + s * stress_center,
                frame=slerp([s]).as_matrix()[0],
                major_semiaxis_m=(
                    (1.0 - s) * float(core_row["major_semiaxis_m"])
                    + s * float(stress_row["major_semiaxis_m"])
                ),
                axis_ratio=(
                    (1.0 - s) * float(core_row["axis_ratio"])
                    + s * float(stress_row["axis_ratio"])
                ),
                source="core01_to_stress00_direct",
                interpolation_s=s,
                generation_seed=0,
            )
        )

    lateral_design = qmc.LatinHypercube(d=6, seed=lateral_seed).random(12)
    center_levels_m = np.asarray([-0.002, -0.001, 0.001, 0.002])
    tilt_levels_deg = np.asarray([-0.5, -0.25, 0.25, 0.5])
    for index, s in enumerate(direct_s):
        base_center = (1.0 - s) * core_center + s * stress_center
        base_frame = slerp([s]).as_matrix()[0]
        sample = lateral_design[index]
        center_delta = np.asarray(
            [
                center_levels_m[min(3, int(sample[j] * 4.0))]
                for j in range(3)
            ]
        )
        tilt_delta = np.deg2rad(
            [
                tilt_levels_deg[min(3, int(sample[j + 3] * 4.0))]
                for j in range(3)
            ]
        )
        center = base_center + base_frame @ center_delta
        frame = Rotation.from_rotvec(base_frame @ tilt_delta).apply(
            base_frame.T
        ).T
        rows.append(
            _catalog_row(
                family_id=f"bridge_lateral_{index:02d}",
                group_id="bridge_lateral",
                order=len(rows),
                center_m=center,
                frame=frame,
                major_semiaxis_m=(
                    (1.0 - s) * float(core_row["major_semiaxis_m"])
                    + s * float(stress_row["major_semiaxis_m"])
                ),
                axis_ratio=(
                    (1.0 - s) * float(core_row["axis_ratio"])
                    + s * float(stress_row["axis_ratio"])
                ),
                source="corridor_lateral_lhs",
                interpolation_s=s,
                generation_seed=lateral_seed,
            )
        )

    lhs = qmc.LatinHypercube(d=8, seed=lhs_seed).random(8)
    low = np.asarray(
        [0.485, 0.325, -0.006, -0.006, -0.003, -2.2, -2.2, -2.2]
    )
    high = np.asarray(
        [0.500, 0.350, 0.006, 0.006, 0.003, 2.2, 2.2, 2.2]
    )
    anchor_center = 0.5 * (core_center + stress_center)
    anchor_frame = slerp([0.5]).as_matrix()[0]
    for index, unit in enumerate(lhs):
        values = low + unit * (high - low)
        frame = Rotation.from_rotvec(
            anchor_frame @ np.deg2rad(values[5:8])
        ).apply(anchor_frame.T).T
        center = anchor_center + anchor_frame @ values[2:5]
        rows.append(
            _catalog_row(
                family_id=f"bridge_lhs_{index:02d}",
                group_id="bridge_lhs",
                order=len(rows),
                center_m=center,
                frame=frame,
                major_semiaxis_m=float(values[0]),
                axis_ratio=float(values[1]),
                source="expanded_box_lhs",
                interpolation_s=None,
                generation_seed=lhs_seed,
            )
        )
    catalog = pd.DataFrame(rows).sort_values("catalog_order", kind="stable")
    if len(catalog) != 32 or not catalog["family_id"].is_unique:
        raise RuntimeError("bridge catalog must contain 32 unique families")
    if catalog["family_fingerprint"].duplicated().any():
        raise RuntimeError("bridge catalog contains duplicate geometry")
    return catalog.reset_index(drop=True)


def cyclic_interpolate_beta(beta_rad: np.ndarray, phase_count: int) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    source = np.arange(len(beta) + 1, dtype=float)
    target = np.arange(int(phase_count), dtype=float) * len(beta) / int(phase_count)
    closed = np.vstack([beta, beta[0]])
    return np.column_stack(
        [np.interp(target, source, closed[:, joint]) for joint in range(6)]
    )


def bridge_seed_beta(
    core_beta_rad: np.ndarray,
    stress_beta_rad: np.ndarray,
    *,
    interpolation_s: float,
    phase_count: int,
) -> np.ndarray:
    core = cyclic_interpolate_beta(core_beta_rad, phase_count)
    stress = cyclic_interpolate_beta(stress_beta_rad, phase_count)
    s = float(interpolation_s)
    return (1.0 - s) * core + s * stress


def tube_targets(
    family: EllipseFamilySpec,
    *,
    phase_count: int,
    u_mm: float,
    v_mm: float,
) -> np.ndarray:
    phase = np.linspace(0.0, 2.0 * np.pi, int(phase_count), endpoint=False)
    target = family.centerline(phase_count=int(phase_count))
    a = float(family.major_semiaxis_m)
    b = float(family.minor_semiaxis_m)
    radial = (
        np.cos(phase)[:, None] / a * family.major_direction[None, :]
        + np.sin(phase)[:, None] / b * family.minor_direction[None, :]
    )
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    return (
        target
        + float(u_mm) / 1000.0 * radial
        + float(v_mm) / 1000.0 * family.plane_normal[None, :]
    )


def row_null_error_frame(
    environment: ExploratoryEnvironment,
    prediction_beta_rad: np.ndarray,
    gold_beta_rad: np.ndarray,
    target_xyz_m: np.ndarray,
    *,
    seed: int,
    family_id: str,
) -> pd.DataFrame:
    prediction = np.asarray(prediction_beta_rad, dtype=float).reshape(-1, 6)
    gold = np.asarray(gold_beta_rad, dtype=float).reshape(-1, 6)
    target = np.asarray(target_xyz_m, dtype=float).reshape(-1, 3)
    if not (len(prediction) == len(gold) == len(target)):
        raise ValueError("diagnostic arrays must have aligned rows")
    achieved = np.asarray(environment.fk(prediction), dtype=float)
    rows: list[dict[str, Any]] = []
    for phase_idx, (pred, truth, xyz, actual) in enumerate(
        zip(prediction, gold, target, achieved)
    ):
        jac = np.asarray(environment.jacobian(truth), dtype=float).reshape(3, 6)
        pinv = np.linalg.pinv(jac, rcond=1.0e-12)
        row_projector = pinv @ jac
        delta = pred - truth
        row_delta = row_projector @ delta
        null_delta = delta - row_delta
        singular = np.linalg.svd(jac, compute_uv=False)
        rows.append(
            {
                "family_id": str(family_id),
                "seed": int(seed),
                "phase_idx": int(phase_idx),
                "total_beta_rms_deg": float(
                    np.rad2deg(np.sqrt(np.mean(np.square(delta))))
                ),
                "row_beta_rms_deg": float(
                    np.rad2deg(np.sqrt(np.mean(np.square(row_delta))))
                ),
                "null_beta_rms_deg": float(
                    np.rad2deg(np.sqrt(np.mean(np.square(null_delta))))
                ),
                "fk_error_mm": float(np.linalg.norm(actual - xyz) * 1000.0),
                "linearized_error_mm": float(
                    np.linalg.norm(jac @ delta) * 1000.0
                ),
                "jacobian_sigma1_m": float(singular[0]),
                "jacobian_sigma2_m": float(singular[1]),
                "jacobian_sigma3_m": float(singular[2]),
                "jacobian_kappa": (
                    math.inf
                    if singular[2] <= 0.0
                    else float(singular[0] / singular[2])
                ),
            }
        )
    return pd.DataFrame(rows)


def dls_refine(
    environment: ExploratoryEnvironment,
    prediction_beta_rad: np.ndarray,
    target_xyz_m: np.ndarray,
    *,
    steps: int,
    damping: float = 1.0e-3,
    weights: Sequence[float] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0),
) -> tuple[np.ndarray, pd.DataFrame]:
    beta = np.asarray(prediction_beta_rad, dtype=float).reshape(-1, 6).copy()
    target = np.asarray(target_xyz_m, dtype=float).reshape(-1, 3)
    if len(beta) != len(target):
        raise ValueError("prediction and target must have aligned rows")
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    updates: list[dict[str, Any]] = []
    for step in range(1, int(steps) + 1):
        achieved = np.asarray(environment.fk(beta), dtype=float)
        next_beta = beta.copy()
        for index in range(len(beta)):
            jac = np.asarray(environment.jacobian(beta[index]), dtype=float)
            update = weighted_damped_pinv(
                jac, damping=float(damping), weights=weights
            ) @ (target[index] - achieved[index])
            next_beta[index] = np.clip(
                beta[index] + update, bounds[:, 0], bounds[:, 1]
            )
            updates.append(
                {
                    "step": int(step),
                    "phase_idx": int(index),
                    "update_rms_deg": float(
                        np.rad2deg(np.sqrt(np.mean(np.square(update))))
                    ),
                }
            )
        beta = next_beta
    return beta, pd.DataFrame(
        updates, columns=["step", "phase_idx", "update_rms_deg"]
    )


def margin_sample_weight(
    minimum_joint_margin_deg: np.ndarray | Sequence[float],
) -> np.ndarray:
    margin = np.asarray(minimum_joint_margin_deg, dtype=float)
    return np.minimum(1.0, np.maximum(0.25, margin / 1.5))


def hard_phase_mask(
    phase_error_mm: np.ndarray | Sequence[float],
    *,
    quantile: float = 0.8,
    cyclic_radius: int = 2,
) -> np.ndarray:
    error = np.asarray(phase_error_mm, dtype=float).reshape(-1)
    selected = error >= float(np.quantile(error, float(quantile)))
    expanded = selected.copy()
    for offset in range(1, int(cyclic_radius) + 1):
        expanded |= np.roll(selected, offset) | np.roll(selected, -offset)
    return expanded


def build_exploratory_model(
    train_xyz_m: np.ndarray,
    *,
    geometry: StudentGeometry,
    architecture: str,
) -> Any:
    units = (
        (512, 512, 256, 128, 64)
        if str(architecture) == "wide"
        else (128, 128, 64)
    )
    return build_static_model(
        np.asarray(train_xyz_m, dtype=np.float32),
        geometry=geometry,
        output_mode="tanh",
        hidden_units=units,
    )


def fine_tune_exploratory_student(
    model: Any,
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    geometry: StudentGeometry,
    seed: int,
    row_loss_weight: float,
    learning_rate: float = 3.0e-4,
    batch_size: int = 256,
    max_steps: int = 4500,
    validation_interval: int = 100,
    patience_intervals: int = 18,
    sampling_weights: Mapping[str, float] | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Fine-tune with weighted beta/FK loss and optional frozen-J row loss."""

    import tensorflow as tf
    from quasi_exp.model.kinematics_tf import forward_xyz_from_beta_tf

    jac_columns = tuple(f"jacobian_{row}_{col}" for row in range(3) for col in range(6))
    required = {*XYZ_COLUMNS, *BETA_COLUMNS, "sample_weight"}
    if float(row_loss_weight) > 0.0:
        required.update(jac_columns)
    for name, frame in (("training", training_frame), ("validation", validation_frame)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} frame missing columns: {missing}")

    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    bounds = tf.constant(
        np.asarray(geometry.beta_bounds_rad, dtype=np.float32), dtype=tf.float32
    )
    beta_scale = tf.constant(
        np.deg2rad(np.asarray([1, 1, 1, 1, 0.75, 1], dtype=np.float32))
    )

    def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
        xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
        beta = frame.loc[:, BETA_COLUMNS].to_numpy(dtype=np.float32)
        weight = frame["sample_weight"].to_numpy(dtype=np.float32)
        if float(row_loss_weight) > 0.0:
            jac = frame.loc[:, jac_columns].to_numpy(dtype=np.float32).reshape(-1, 3, 6)
        else:
            jac = np.zeros((len(frame), 3, 6), dtype=np.float32)
        return xyz, beta, weight, jac

    train_arrays = arrays(training_frame)
    valid_arrays = arrays(validation_frame)
    if "sampling_bucket" in training_frame:
        observed = tuple(
            str(value)
            for value in training_frame["sampling_bucket"].drop_duplicates()
        )
        if sampling_weights is not None:
            unknown = sorted(set(sampling_weights) - set(observed))
            missing = sorted(set(observed) - set(sampling_weights))
            if unknown or missing:
                raise ValueError(
                    "sampling_weights must exactly cover observed buckets; "
                    f"unknown={unknown}, missing={missing}"
                )
            available = list(sampling_weights)
            bucket_weights = [float(sampling_weights[name]) for name in available]
            values = np.asarray(bucket_weights, dtype=float)
            if (
                not np.isfinite(values).all()
                or np.any(values <= 0.0)
                or not np.isclose(values.sum(), 1.0)
            ):
                raise ValueError(
                    "sampling_weights must be finite, positive and sum to one"
                )
        else:
            available = [
                bucket
                for bucket in ("core", "bridge", "boundary")
                if bucket in observed
            ]
            if set(available) != set(observed):
                raise ValueError(
                    f"unsupported implicit sampling buckets: {sorted(observed)}"
                )
            if available == ["core", "bridge", "boundary"]:
                bucket_weights = [0.4, 0.4, 0.2]
            elif available == ["core", "boundary"]:
                bucket_weights = [0.8, 0.2]
            elif available == ["core", "bridge"]:
                bucket_weights = [0.4, 0.6]
            else:
                bucket_weights = [1.0 / len(available)] * len(available)
        sources = []
        for offset, bucket in enumerate(available):
            selected = training_frame["sampling_bucket"].eq(bucket).to_numpy()
            bucket_arrays = tuple(value[selected] for value in train_arrays)
            sources.append(
                tf.data.Dataset.from_tensor_slices(bucket_arrays)
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
        sampler = {
            bucket: float(weight)
            for bucket, weight in zip(available, bucket_weights, strict=True)
        }
    else:
        dataset = (
            tf.data.Dataset.from_tensor_slices(train_arrays)
            .shuffle(
                len(training_frame),
                seed=int(seed),
                reshuffle_each_iteration=True,
            )
            .repeat()
        )
        sampler = {"all": 1.0}
    dataset = dataset.batch(
        int(batch_size), drop_remainder=False
    ).prefetch(tf.data.AUTOTUNE)
    iterator = iter(dataset)
    optimizer = tf.keras.optimizers.Adam(float(learning_rate))

    def loss_terms(batch: tuple[Any, ...], training: bool) -> tuple[Any, ...]:
        xyz, beta_true, sample_weight, jac = batch
        pred = model(xyz, training=training)
        beta_point = tf.reduce_mean(tf.square((pred - beta_true) / beta_scale), axis=1)
        pred_xyz = forward_xyz_from_beta_tf(
            pred,
            lengths_m=geometry.lengths_m,
            p_end_local_m=geometry.p_end_local_m,
            theta_sign=geometry.theta_sign,
        )
        fk_point = tf.reduce_mean(
            tf.square((pred_xyz - xyz) / tf.cast(0.003, pred.dtype)), axis=1
        )
        margin = tf.minimum(pred - bounds[:, 0], bounds[:, 1] - pred)
        margin_point = tf.reduce_mean(
            tf.square(
                tf.nn.relu(
                    (tf.cast(np.deg2rad(1.5), pred.dtype) - margin)
                    / tf.cast(np.deg2rad(0.5), pred.dtype)
                )
            ),
            axis=1,
        )
        projected = tf.einsum("nij,nj->ni", jac, pred - beta_true)
        row_point = tf.reduce_mean(
            tf.square(projected / tf.cast(0.003, pred.dtype)), axis=1
        )
        point = (
            beta_point
            + tf.cast(0.5, pred.dtype) * fk_point
            + tf.cast(0.5, pred.dtype) * margin_point
            + tf.cast(row_loss_weight, pred.dtype) * row_point
        )
        total = tf.reduce_sum(point * sample_weight) / tf.reduce_sum(sample_weight)
        return total, tf.reduce_mean(beta_point), tf.reduce_mean(fk_point), tf.reduce_mean(row_point)

    best = math.inf
    best_weights = model.get_weights()
    stale = 0
    rows: list[dict[str, Any]] = []
    valid_tensor = tuple(tf.convert_to_tensor(value) for value in valid_arrays)
    for step in range(1, int(max_steps) + 1):
        batch = next(iterator)
        with tf.GradientTape() as tape:
            objective, _beta, _fk, _row = loss_terms(batch, True)
        gradients = tape.gradient(objective, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if step % int(validation_interval) != 0 and step != int(max_steps):
            continue
        valid_loss, beta_loss, fk_loss, row_loss = loss_terms(valid_tensor, False)
        value = float(valid_loss.numpy())
        rows.append(
            {
                "step": int(step),
                "validation_objective": value,
                "validation_beta_loss": float(beta_loss.numpy()),
                "validation_fk_loss": float(fk_loss.numpy()),
                "validation_row_loss": float(row_loss.numpy()),
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
    history = pd.DataFrame(rows)
    return model, history, {
        "seed": int(seed),
        "training_rows": int(len(training_frame)),
        "validation_rows": int(len(validation_frame)),
        "row_loss_weight": float(row_loss_weight),
        "optimizer_steps": int(history["step"].max()) if len(history) else 0,
        "best_validation_objective": float(best),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "sampling_weights": sampler,
    }
