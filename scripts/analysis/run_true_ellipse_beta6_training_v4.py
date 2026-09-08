#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from features import build_features  # noqa: E402
from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from true_ellipse_atlas_utils import beta_bounds_rad, theta_from_beta_batch  # noqa: E402
from true_ellipse_radial_bundle_engine import (  # noqa: E402
    JointDomainSpec,
    joint_margin_report,
    registered_joint_domain,
)


TARGET_XYZ_COLS = ["x_target_m", "y_target_m", "z_target_m"]
FK_XYZ_COLS = ["x_m", "y_m", "z_m"]
BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
DEFAULT_TUBE = REPO_ROOT / "runs" / "true_ellipse_branch_lifting_v3" / "06_local_tube" / "tube_small.parquet"
DEFAULT_OUT = REPO_ROOT / "runs" / "true_ellipse_beta6_training_v4"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml"
DEFAULT_SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)
VALIDATION_OFFSETS = {(-2.5, 0.0), (2.5, 0.0), (0.0, -2.5), (0.0, 2.5)}


@dataclass(frozen=True)
class TubeSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    hidden_layers: tuple[int, ...]
    feature_set: str
    activation: str = "relu"
    alpha: float = 1.0e-6

    @property
    def config_id(self) -> str:
        alpha = f"{self.alpha:.0e}".replace("-", "m").replace("+", "p")
        return f"{self.architecture}_{self.feature_set}_{self.activation}_a{alpha}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(v) for v in value]


def parse_phases(value: str) -> list[str]:
    phases = [part.strip() for part in str(value).split(",") if part.strip()]
    if phases == ["all"]:
        return ["audit", "split", "screen", "train", "sweep", "visualize", "summary"]
    valid = {"audit", "split", "screen", "train", "sweep", "visualize", "summary"}
    unknown = sorted(set(phases) - valid)
    if unknown:
        raise ValueError(f"unsupported phases: {unknown}")
    return phases


def generate_exact_ellipse(metadata: dict[str, Any] | pd.Series, *, amp_xy_mm: float, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False)
    amp_xy = float(amp_xy_mm) / 1000.0
    amp_z = 1.5 * amp_xy
    xyz = np.column_stack(
        [
            float(metadata["center_x_m"]) + amp_xy * np.sin(t),
            float(metadata["center_y_m"]) + amp_xy * np.sin(t + float(metadata["phase_y_rad"])),
            float(metadata["center_z_m"]) + amp_z * np.sin(t + float(metadata["phase_z_rad"])),
        ]
    )
    return xyz, t


def make_tube_split(df: pd.DataFrame) -> TubeSplit:
    n1 = df["delta_n1_mm"].to_numpy(dtype=float)
    n2 = df["delta_n2_mm"].to_numpy(dtype=float)
    test_mask = np.isclose(n1, 0.0) & np.isclose(n2, 0.0)
    val_mask = np.zeros(len(df), dtype=bool)
    for v1, v2 in VALIDATION_OFFSETS:
        val_mask |= np.isclose(n1, v1) & np.isclose(n2, v2)
    val_mask &= ~test_mask
    train_mask = ~(test_mask | val_mask)
    train_idx = np.flatnonzero(train_mask).astype(np.int64)
    val_idx = np.flatnonzero(val_mask).astype(np.int64)
    test_idx = np.flatnonzero(test_mask).astype(np.int64)
    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise ValueError("tube split produced an empty partition")
    if len(set(train_idx) & set(val_idx)) or len(set(train_idx) & set(test_idx)) or len(set(val_idx) & set(test_idx)):
        raise AssertionError("tube split leaked rows between partitions")
    return TubeSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def compute_support_metrics(training_xyz: np.ndarray, target_xyz: np.ndarray, *, radius_mm: float = 15.0) -> dict[str, Any]:
    train = np.asarray(training_xyz, dtype=float).reshape(-1, 3)
    target = np.asarray(target_xyz, dtype=float).reshape(-1, 3)
    if len(train) == 0 or len(target) == 0:
        raise ValueError("support metrics require non-empty training and target arrays")
    tree = cKDTree(train)
    nn = np.asarray(tree.query(target, k=1)[0], dtype=float) * 1000.0
    counts = np.asarray([len(ids) for ids in tree.query_ball_point(target, float(radius_mm) / 1000.0)], dtype=float)
    return {
        "nn_mean_mm": float(np.mean(nn)),
        "nn_p50_mm": float(np.percentile(nn, 50)),
        "nn_p95_mm": float(np.percentile(nn, 95)),
        "nn_max_mm": float(np.max(nn)),
        "tube_count_min": int(np.min(counts)),
        "tube_count_p10": float(np.percentile(counts, 10)),
        "tube_count_p50": float(np.percentile(counts, 50)),
        "tube_count_p95": float(np.percentile(counts, 95)),
    }


def _beta_rms_deg(delta_rad: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta_rad, dtype=float)
    return np.sqrt(np.mean(np.square(delta), axis=-1)) * 180.0 / math.pi


def periodic_beta_metrics(beta_rad: np.ndarray) -> dict[str, float]:
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    if len(beta) < 3:
        raise ValueError("periodic metrics require at least three samples")
    delta = np.roll(beta, -1, axis=0) - beta
    delta2 = np.roll(beta, -1, axis=0) - 2.0 * beta + np.roll(beta, 1, axis=0)
    d = _beta_rms_deg(delta)
    d2 = _beta_rms_deg(delta2)
    seam = float(_beta_rms_deg(beta[:1] - beta[-1:])[0])
    return {
        "delta_beta_p95_deg": float(np.percentile(d, 95)),
        "delta_beta_max_deg": float(np.max(d)),
        "delta2_beta_p95_deg": float(np.percentile(d2, 95)),
        "delta2_beta_max_deg": float(np.max(d2)),
        "seam_beta_rms_deg": seam,
    }


def linear_beta_metrics(beta_rad: np.ndarray) -> dict[str, float | None]:
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    if len(beta) < 3:
        raise ValueError("linear metrics require at least three samples")
    delta = np.diff(beta, axis=0)
    delta2 = np.diff(beta, n=2, axis=0)
    d = _beta_rms_deg(delta)
    d2 = _beta_rms_deg(delta2)
    return {
        "delta_beta_p95_deg": float(np.percentile(d, 95)),
        "delta_beta_max_deg": float(np.max(d)),
        "delta2_beta_p95_deg": float(np.percentile(d2, 95)),
        "delta2_beta_max_deg": float(np.max(d2)),
        "seam_beta_rms_deg": None,
    }


def grouped_periodic_beta_metrics(beta_rad: np.ndarray, groups: np.ndarray, angle: np.ndarray) -> dict[str, float]:
    frames = []
    group_arr = np.asarray(groups)
    angle_arr = np.asarray(angle, dtype=float)
    for group in pd.unique(group_arr):
        idx = np.flatnonzero(group_arr == group)
        idx = idx[np.argsort(angle_arr[idx])]
        if len(idx) >= 3:
            frames.append(periodic_beta_metrics(np.asarray(beta_rad)[idx]))
    if not frames:
        raise ValueError("no periodic group with at least three samples")
    return {
        "delta_beta_p95_deg": float(max(row["delta_beta_p95_deg"] for row in frames)),
        "delta_beta_max_deg": float(max(row["delta_beta_max_deg"] for row in frames)),
        "delta2_beta_p95_deg": float(max(row["delta2_beta_p95_deg"] for row in frames)),
        "delta2_beta_max_deg": float(max(row["delta2_beta_max_deg"] for row in frames)),
        "seam_beta_rms_deg": float(max(row["seam_beta_rms_deg"] for row in frames)),
    }


def axis_error_metrics(target_xyz: np.ndarray, achieved_xyz: np.ndarray) -> dict[str, Any]:
    err = (np.asarray(achieved_xyz, dtype=float) - np.asarray(target_xyz, dtype=float)) * 1000.0
    out: dict[str, Any] = {}
    p95s = []
    fixed = []
    for axis_idx, axis in enumerate("xyz"):
        vals = err[:, axis_idx]
        p95 = float(np.percentile(np.abs(vals), 95))
        out[f"{axis}err_mean_mm"] = float(np.mean(vals))
        out[f"{axis}err_p95_abs_mm"] = p95
        out[f"{axis}err_same_sign_ratio"] = float(max(np.mean(vals > 0.0), np.mean(vals < 0.0)))
        out[f"fixed_{axis}_bias_gt2mm"] = bool(np.all(vals > 2.0) or np.all(vals < -2.0))
        p95s.append(p95)
        fixed.append(out[f"fixed_{axis}_bias_gt2mm"])
    out["axiserr_max_p95_abs_mm"] = float(max(p95s))
    out["fixed_any_axis_bias_gt2mm"] = bool(any(fixed))
    return out


def evaluate_beta_prediction(
    *,
    beta_pred: np.ndarray,
    target_xyz: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    beta_true: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    angle: np.ndarray | None = None,
    periodic: bool = True,
    joint_domain: JointDomainSpec | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    beta = np.asarray(beta_pred, dtype=float).reshape(-1, 6)
    target = np.asarray(target_xyz, dtype=float).reshape(-1, 3)
    theta = theta_from_beta_batch(beta, theta_sign=float(theta_sign))
    achieved = fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    ee = np.linalg.norm(achieved - target, axis=1) * 1000.0
    out: dict[str, Any] = {
        "rows": int(len(beta)),
        "ee_mean_mm": float(np.mean(ee)),
        "ee_p95_mm": float(np.percentile(ee, 95)),
        "ee_max_mm": float(np.max(ee)),
    }
    out.update(axis_error_metrics(target, achieved))
    if beta_true is not None:
        beta_err = np.abs(np.asarray(beta_true, dtype=float).reshape(-1, 6) - beta) * 180.0 / math.pi
        out["beta_mae_deg"] = float(np.mean(beta_err))
        out["beta_p95_deg"] = float(np.percentile(beta_err, 95))
        out["theta_mae_deg"] = out["beta_mae_deg"]
        out["theta_p95_deg"] = out["beta_p95_deg"]
    if not bool(periodic):
        out.update(linear_beta_metrics(beta))
    elif groups is not None and angle is not None:
        out.update(grouped_periodic_beta_metrics(beta, np.asarray(groups), np.asarray(angle)))
    else:
        out.update(periodic_beta_metrics(beta))
    domain = registered_joint_domain("current_v6") if joint_domain is None else joint_domain
    margin = joint_margin_report(beta, domain=domain)
    out["joint_domain_id"] = domain.domain_id
    out["joint_domain_fingerprint"] = domain.fingerprint
    out["prediction_min_joint_margin_deg"] = float(margin["min_joint_margin_deg"])
    out["prediction_joint_margin_p01_deg"] = float(margin["joint_margin_p01_deg"])
    out["prediction_joint_margin_p05_deg"] = float(margin["joint_margin_p05_deg"])
    out["beta_bound_violation_count"] = int(margin["out_of_bounds_count"])
    out["beta_bound_violation_ratio"] = float(margin["out_of_bounds_count"] / max(1, beta.size))
    return out, achieved, theta


def centerline_model_gate(metrics: dict[str, Any], *, require_beta_error: bool = True) -> bool:
    beta_ok = True if not require_beta_error else float(metrics.get("beta_p95_deg", np.inf)) <= 1.0
    return bool(
        beta_ok
        and float(metrics.get("ee_p95_mm", np.inf)) <= 5.0
        and float(metrics.get("ee_max_mm", np.inf)) <= 10.0
        and float(metrics.get("axiserr_max_p95_abs_mm", np.inf)) <= 3.0
        and not bool(metrics.get("fixed_any_axis_bias_gt2mm", True))
        and float(metrics.get("delta_beta_p95_deg", np.inf)) <= 1.0
        and float(metrics.get("delta_beta_max_deg", np.inf)) <= 2.0
        and float(metrics.get("delta2_beta_p95_deg", np.inf)) <= 0.25
        and float(metrics.get("seam_beta_rms_deg", np.inf)) <= 1.0
        and int(metrics.get("beta_bound_violation_count", 1)) == 0
    )


def aggregate_seed_gate(rows: pd.DataFrame, *, required_fraction: float = 0.8) -> dict[str, Any]:
    total = int(len(rows))
    passed = int(rows["model_gate_pass"].astype(bool).sum()) if total else 0
    required = int(math.ceil(float(required_fraction) * total)) if total else 1
    return {
        "passed_seed_count": passed,
        "total_seed_count": total,
        "required_seed_count": required,
        "seed_pass_fraction": float(passed / total) if total else 0.0,
        "stable_gate_pass": bool(total > 0 and passed >= required),
    }


def contiguous_max_radius(table: pd.DataFrame, gate_col: str, *, anchor_mm: float = 75.0) -> float | None:
    rows = table.sort_values("amp_xy_mm").reset_index(drop=True)
    rows = rows[rows["amp_xy_mm"].to_numpy(dtype=float) >= float(anchor_mm) - 1.0e-9]
    if rows.empty or not np.isclose(float(rows.iloc[0]["amp_xy_mm"]), float(anchor_mm), atol=1.0e-8):
        return None
    maximum: float | None = None
    for _idx, row in rows.iterrows():
        if not bool(row[gate_col]):
            break
        maximum = float(row["amp_xy_mm"])
    return maximum


def select_stable_radii(table: pd.DataFrame, *, anchor_mm: float = 75.0) -> dict[str, float | None]:
    return {
        "strict_supported_rmax_mm": contiguous_max_radius(table, "strict_gate_pass", anchor_mm=anchor_mm),
        "relaxed_supported_rmax_mm": contiguous_max_radius(table, "relaxed_gate_pass", anchor_mm=anchor_mm),
        "model_only_rmax_mm": contiguous_max_radius(table, "model_only_gate_pass", anchor_mm=anchor_mm),
    }


def classify_strict_limit(table: pd.DataFrame, *, strict_rmax_mm: float | None) -> str:
    if strict_rmax_mm is None:
        return "anchor_or_model_gate"
    rows = table.sort_values("amp_xy_mm").reset_index(drop=True)
    later = rows[rows["amp_xy_mm"].to_numpy(dtype=float) > float(strict_rmax_mm) + 1.0e-9]
    if later.empty:
        return "search_grid_upper_bound"
    next_row = later.iloc[0]
    if bool(next_row["stable_gate_pass"]) and not bool(next_row["strict_support_gate_pass"]):
        return "data_support"
    if not bool(next_row["stable_gate_pass"]):
        return "model_stability"
    return "combined_gate"


def model_configs(*, activation: str = "relu", alphas: Iterable[float] = (1.0e-6,), limited: list[tuple[str, str]] | None = None) -> list[ModelConfig]:
    architectures = {
        "mlp_beta6": (256, 128, 64, 32),
        "mlp_beta6_large": (512, 256, 128, 64),
    }
    configs = []
    for architecture, hidden in architectures.items():
        for feature_set in ("raw", "poly_medium", "poly_heavy"):
            if limited is not None and (architecture, feature_set) not in limited:
                continue
            for alpha in alphas:
                configs.append(
                    ModelConfig(
                        architecture=architecture,
                        hidden_layers=hidden,
                        feature_set=feature_set,
                        activation=str(activation),
                        alpha=float(alpha),
                    )
                )
    return configs


def config_from_dict(data: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        architecture=str(data["architecture"]),
        hidden_layers=tuple(int(v) for v in data["hidden_layers"]),
        feature_set=str(data["feature_set"]),
        activation=str(data.get("activation", "relu")),
        alpha=float(data.get("alpha", 1.0e-6)),
    )


def make_model(config: ModelConfig, *, seed: int, max_iter: int, batch_size: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=tuple(config.hidden_layers),
        activation=str(config.activation),
        solver="adam",
        alpha=float(config.alpha),
        batch_size=int(batch_size),
        learning_rate_init=1.0e-3,
        max_iter=int(max_iter),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=40,
        random_state=int(seed),
        verbose=False,
    )


def _fit_scaled_model(
    *,
    config: ModelConfig,
    seed: int,
    xyz_train: np.ndarray,
    beta_train: np.ndarray,
    max_iter: int,
    batch_size: int,
) -> tuple[MLPRegressor, StandardScaler, StandardScaler, list[str], float]:
    features, names = build_features(np.asarray(xyz_train, dtype=float), feature_set=config.feature_set)
    x_scaler = StandardScaler().fit(features)
    y_scaler = StandardScaler().fit(np.asarray(beta_train, dtype=float))
    model = make_model(config, seed=seed, max_iter=max_iter, batch_size=min(int(batch_size), max(1, len(features))))
    start = time.perf_counter()
    model.fit(x_scaler.transform(features), y_scaler.transform(beta_train))
    fit_s = float(time.perf_counter() - start)
    return model, x_scaler, y_scaler, names, fit_s


def predict_beta(package: dict[str, Any], xyz: np.ndarray) -> np.ndarray:
    features, _names = build_features(np.asarray(xyz, dtype=float), feature_set=str(package["feature_set"]))
    pred_scaled = package["model"].predict(package["x_scaler"].transform(features))
    return np.asarray(package["y_scaler"].inverse_transform(pred_scaled), dtype=float).reshape(-1, 6)


def tube_metadata(df: pd.DataFrame) -> dict[str, Any]:
    fields = [
        "candidate_id",
        "ellipse_id",
        "center_x_m",
        "center_y_m",
        "center_z_m",
        "amp_xy_mm",
        "amp_z_mm",
        "phase_y_rad",
        "phase_z_rad",
    ]
    out: dict[str, Any] = {}
    for field in fields:
        values = pd.unique(df[field])
        if len(values) != 1:
            raise ValueError(f"expected one {field}, got {len(values)}")
        value = values[0]
        out[field] = value.item() if isinstance(value, np.generic) else value
    return out


def audit_tube_dataset(df: pd.DataFrame, *, theta_sign: float) -> dict[str, Any]:
    required = {
        *TARGET_XYZ_COLS,
        *FK_XYZ_COLS,
        *BETA_COLS,
        *THETA_COLS,
        "candidate_id",
        "ellipse_id",
        "angle_idx",
        "angle_rad",
        "delta_n1_mm",
        "delta_n2_mm",
        "tube_offset_id",
        "is_centerline",
        "ik_success",
        "tube_success",
        "xyz_residual_mm",
        "continuation_run_id",
        "phase_y_rad",
        "phase_z_rad",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"tube dataset missing columns: {missing}")
    duplicate_count = int(df.duplicated(["angle_idx", "tube_offset_id"]).sum())
    beta = df[BETA_COLS].to_numpy(dtype=float)
    theta_expected = theta_from_beta_batch(beta, theta_sign=float(theta_sign))
    theta_actual = df[THETA_COLS].to_numpy(dtype=float)
    target = df[TARGET_XYZ_COLS].to_numpy(dtype=float)
    achieved = df[FK_XYZ_COLS].to_numpy(dtype=float)
    reconstruction_error = np.abs(theta_expected - theta_actual)
    xyz_gap = np.linalg.norm(target - achieved, axis=1) * 1000.0
    metadata = tube_metadata(df)
    report: dict[str, Any] = {
        "rows": int(len(df)),
        "unique_angles": int(df["angle_idx"].nunique()),
        "unique_offsets": int(df["tube_offset_id"].nunique()),
        "duplicate_angle_offset_rows": duplicate_count,
        "ik_success_ratio": float(df["ik_success"].astype(bool).mean()),
        "tube_success_ratio": float(df["tube_success"].astype(bool).mean()),
        "theta_reconstruction_max_abs_rad": float(np.max(reconstruction_error)),
        "target_fk_gap_p95_mm": float(np.percentile(xyz_gap, 95)),
        "target_fk_gap_max_mm": float(np.max(xyz_gap)),
        "reported_residual_p95_mm": float(np.percentile(df["xyz_residual_mm"].to_numpy(dtype=float), 95)),
        "continuation_run_ids": sorted(df["continuation_run_id"].astype(str).unique().tolist()),
        "metadata": metadata,
    }
    checks = {
        "rows_9000": report["rows"] == 9000,
        "angles_360": report["unique_angles"] == 360,
        "offsets_25": report["unique_offsets"] == 25,
        "no_duplicate_keys": duplicate_count == 0,
        "all_ik_success": report["ik_success_ratio"] == 1.0,
        "all_tube_success": report["tube_success_ratio"] == 1.0,
        "theta_reconstruction": report["theta_reconstruction_max_abs_rad"] <= 1.0e-12,
        "selected_branch_only": (
            len(report["continuation_run_ids"]) == 1
            and report["continuation_run_ids"][0].strip().lower() not in {"", "nan", "none"}
        ),
    }
    report["checks"] = checks
    report["audit_gate_pass"] = bool(all(checks.values()))
    return report


def _subset_by_stride(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    stride = max(1, int(stride))
    if stride == 1:
        return df.reset_index(drop=True)
    return df[df["angle_idx"].to_numpy(dtype=int) % stride == 0].reset_index(drop=True)


def _task_mode_indices(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    split = make_tube_split(df)
    if mode == "screen":
        return split.train_idx, split.val_idx
    if mode == "final":
        train_idx = np.flatnonzero(~df["is_centerline"].astype(bool).to_numpy()).astype(np.int64)
        return train_idx, split.test_idx
    if mode == "sector":
        angle_deg = np.mod(np.rad2deg(df["angle_rad"].to_numpy(dtype=float)), 360.0)
        sector = (angle_deg >= 330.0) & (angle_deg < 360.0)
        noncenter = ~df["is_centerline"].astype(bool).to_numpy()
        train_idx = np.flatnonzero(noncenter & ~sector).astype(np.int64)
        eval_idx = np.flatnonzero(df["is_centerline"].astype(bool).to_numpy() & sector).astype(np.int64)
        return train_idx, eval_idx
    if mode == "full_diagnostic":
        return np.arange(len(df), dtype=np.int64), split.test_idx
    raise ValueError(f"unsupported task mode: {mode}")


def run_training_worker(task: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(task["dataset"])
    df = _subset_by_stride(pd.read_parquet(dataset_path), int(task.get("angle_stride", 1)))
    train_idx, eval_idx = _task_mode_indices(df, str(task["mode"]))
    config = config_from_dict(dict(task["config"]))
    seed = int(task["seed"])
    cfg = load_config(str(task["robot_config"]))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    model, x_scaler, y_scaler, feature_names, fit_s = _fit_scaled_model(
        config=config,
        seed=seed,
        xyz_train=df.iloc[train_idx][TARGET_XYZ_COLS].to_numpy(dtype=float),
        beta_train=df.iloc[train_idx][BETA_COLS].to_numpy(dtype=float),
        max_iter=int(task["max_iter"]),
        batch_size=int(task["batch_size"]),
    )
    package: dict[str, Any] = {
        "kind": "beta6_pose",
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "input_cols": TARGET_XYZ_COLS,
        "feature_set": config.feature_set,
        "feature_names": feature_names,
        "target_cols": BETA_COLS,
        "beta_cols": BETA_COLS,
        "theta_cols": THETA_COLS,
        "theta_sign": theta_sign,
        "robot_config": str(task["robot_config"]),
        "dataset": str(dataset_path),
        "split_metadata": {
            "mode": str(task["mode"]),
            "centerline_held_out": str(task["mode"]) != "full_diagnostic",
            "angle_stride": int(task.get("angle_stride", 1)),
            "train_rows": int(len(train_idx)),
            "eval_rows": int(len(eval_idx)),
        },
        "seed": seed,
        "model_name": config.architecture,
        "activation": config.activation,
        "alpha": config.alpha,
    }
    eval_xyz = df.iloc[eval_idx][TARGET_XYZ_COLS].to_numpy(dtype=float)
    beta_true = df.iloc[eval_idx][BETA_COLS].to_numpy(dtype=float)
    beta_pred = predict_beta(package, eval_xyz)
    is_sector = str(task["mode"]) == "sector"
    eval_metrics, achieved, theta_pred = evaluate_beta_prediction(
        beta_pred=beta_pred,
        beta_true=beta_true,
        target_xyz=eval_xyz,
        lengths_m=inputs.lengths_m,
        p_end_local_m=inputs.p_end_local_m,
        theta_sign=theta_sign,
        groups=None if is_sector else df.iloc[eval_idx]["tube_offset_id"].to_numpy(),
        angle=None if is_sector else df.iloc[eval_idx]["angle_rad"].to_numpy(dtype=float),
        periodic=not is_sector,
    )
    eval_metrics.update(
        {
            "fit_s": fit_s,
            "n_iter": int(model.n_iter_),
            "loss": float(model.loss_),
            "train_rows": int(len(train_idx)),
            "eval_rows": int(len(eval_idx)),
        }
    )
    if str(task["mode"]) == "screen":
        eval_metrics["screen_gate_pass"] = bool(
            float(eval_metrics.get("ee_p95_mm", np.inf)) <= 3.0
            and float(eval_metrics.get("beta_p95_deg", np.inf)) <= 0.5
        )
        eval_metrics["model_gate_pass"] = bool(eval_metrics["screen_gate_pass"])
    elif is_sector:
        eval_metrics["model_gate_pass"] = bool(
            float(eval_metrics.get("ee_p95_mm", np.inf)) <= 5.0
            and float(eval_metrics.get("beta_p95_deg", np.inf)) <= 1.0
            and float(eval_metrics.get("axiserr_max_p95_abs_mm", np.inf)) <= 3.0
            and float(eval_metrics.get("delta_beta_max_deg", np.inf)) <= 2.0
            and float(eval_metrics.get("delta2_beta_p95_deg", np.inf)) <= 0.25
            and int(eval_metrics.get("beta_bound_violation_count", 1)) == 0
        )
    else:
        eval_metrics["model_gate_pass"] = centerline_model_gate(eval_metrics, require_beta_error=True)

    package_path = Path(task["package_path"]) if task.get("package_path") else None
    if package_path is not None:
        package_path.parent.mkdir(parents=True, exist_ok=True)
        dump(package, package_path)
    prediction_path = Path(task["prediction_path"]) if task.get("prediction_path") else None
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pred = df.iloc[eval_idx][
            [
                "candidate_id",
                "ellipse_id",
                "angle_idx",
                "angle_rad",
                "tube_offset_id",
                "delta_n1_mm",
                "delta_n2_mm",
                *TARGET_XYZ_COLS,
            ]
        ].copy()
        for i, col in enumerate(BETA_COLS):
            pred[f"true_{col}"] = beta_true[:, i]
            pred[f"pred_{col}"] = beta_pred[:, i]
        for i, axis in enumerate("xyz"):
            pred[f"achieved_{axis}_m"] = achieved[:, i]
            pred[f"{axis}_err_mm"] = (achieved[:, i] - eval_xyz[:, i]) * 1000.0
        pred["ee_err_mm"] = np.linalg.norm(achieved - eval_xyz, axis=1) * 1000.0
        pred.to_parquet(prediction_path, index=False, compression="zstd")

    result = {
        "task_id": str(task["task_id"]),
        "mode": str(task["mode"]),
        "seed": seed,
        "config_id": config.config_id,
        "config": asdict(config),
        "metrics": eval_metrics,
        "package_path": str(package_path) if package_path is not None else None,
        "prediction_path": str(prediction_path) if prediction_path is not None else None,
    }
    result_path = Path(task["result_path"])
    write_json(result_path, result)
    return result


def _run_worker_subprocess(task_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[name] = "1"
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-task", str(task_path)],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed for {task_path.name}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    task = read_json(task_path)
    result_path = Path(task["result_path"])
    if not result_path.exists():
        raise RuntimeError(f"worker did not create result: {result_path}")
    return read_json(result_path)


def run_tasks(tasks: list[dict[str, Any]], *, task_dir: Path, workers: int, skip_existing: bool) -> list[dict[str, Any]]:
    task_dir.mkdir(parents=True, exist_ok=True)
    task_paths: list[Path] = []
    finished: list[dict[str, Any]] = []
    for task in tasks:
        path = task_dir / f"{task['task_id']}.json"
        result_path = Path(task["result_path"])
        if skip_existing and result_path.exists():
            finished.append(read_json(result_path))
            continue
        write_json(path, task)
        task_paths.append(path)
    if task_paths:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = {executor.submit(_run_worker_subprocess, path): path for path in task_paths}
            for future in as_completed(futures):
                finished.append(future.result())
    return sorted(finished, key=lambda row: str(row["task_id"]))


def preset_settings(preset: str) -> dict[str, Any]:
    preset = str(preset)
    if preset == "smoke":
        return {"angle_stride": 30, "max_iter": 30, "screen_seed_count": 1, "final_seed_count": 1, "screen_config_limit": 2}
    if preset == "pilot":
        return {"angle_stride": 5, "max_iter": 300, "screen_seed_count": 3, "final_seed_count": 3, "screen_config_limit": 4}
    if preset == "formal":
        return {"angle_stride": 1, "max_iter": 800, "screen_seed_count": 3, "final_seed_count": 5, "screen_config_limit": 0}
    raise ValueError(f"unsupported preset: {preset}")


def phase_audit(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "00_input_audit"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.robot_config))
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    df = pd.read_parquet(args.tube_dataset)
    report = audit_tube_dataset(df, theta_sign=theta_sign)
    report["dataset"] = str(args.tube_dataset)
    report["theta_sign"] = theta_sign
    write_json(out / "audit_report.json", report)
    pd.DataFrame(
        [
            {
                "dataset": str(args.tube_dataset),
                "rows": len(df),
                "columns": len(df.columns),
                "unique_angles": df["angle_idx"].nunique(),
                "unique_offsets": df["tube_offset_id"].nunique(),
                "audit_gate_pass": report["audit_gate_pass"],
            }
        ]
    ).to_csv(out / "input_manifest.csv", index=False)
    if not bool(report["audit_gate_pass"]):
        raise RuntimeError(f"V4 input audit failed: {report['checks']}")
    return report


def _radius_values(args: argparse.Namespace, *, preset: str | None = None) -> np.ndarray:
    if preset == "smoke":
        return np.asarray([75.0], dtype=float)
    if preset == "pilot":
        return np.arange(70.0, 85.0 + 1.0e-9, 2.5, dtype=float)
    start = float(args.radius_min_mm)
    stop = float(args.radius_max_mm)
    step = float(args.radius_step_mm)
    values = np.arange(start, stop + step * 0.25, step, dtype=float)
    values = np.round(values, 8)
    if not np.any(np.isclose(values, 75.0, atol=1.0e-8)):
        values = np.unique(np.append(values, 75.0))
    return np.sort(values)


def phase_split(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "01_split_and_support"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.tube_dataset)
    split = make_tube_split(df)
    assignment = df[
        ["angle_idx", "angle_rad", "tube_offset_id", "delta_n1_mm", "delta_n2_mm", "is_centerline", *TARGET_XYZ_COLS]
    ].copy()
    assignment["split"] = "train"
    assignment.loc[split.val_idx, "split"] = "validation"
    assignment.loc[split.test_idx, "split"] = "test_centerline"
    assignment.to_parquet(out / "split_assignment.parquet", index=False, compression="zstd")
    final_train_idx = np.flatnonzero(~df["is_centerline"].astype(bool).to_numpy())
    training_xyz = df.iloc[final_train_idx][TARGET_XYZ_COLS].to_numpy(dtype=float)
    metadata = tube_metadata(df)
    rows = []
    for amp in _radius_values(args):
        target, _angle = generate_exact_ellipse(metadata, amp_xy_mm=float(amp), n_points=360)
        support = compute_support_metrics(training_xyz, target, radius_mm=15.0)
        strict = bool(support["nn_p95_mm"] <= 5.0 and support["nn_max_mm"] <= 8.0 and support["tube_count_p10"] >= 32.0)
        relaxed = bool(support["nn_p95_mm"] <= 8.0 and support["tube_count_p10"] >= 16.0)
        rows.append({"amp_xy_mm": float(amp), "amp_z_mm": 1.5 * float(amp), **support, "strict_support_gate_pass": strict, "relaxed_support_gate_pass": relaxed})
    support_df = pd.DataFrame(rows)
    support_df.to_csv(out / "radius_support_prescan.csv", index=False)
    report = {
        "train_rows_screen": int(len(split.train_idx)),
        "validation_rows": int(len(split.val_idx)),
        "test_centerline_rows": int(len(split.test_idx)),
        "final_training_rows": int(len(final_train_idx)),
        "centerline_overlap_with_train": int(np.count_nonzero(np.isin(split.test_idx, final_train_idx))),
        "validation_offsets": sorted([list(pair) for pair in VALIDATION_OFFSETS]),
        "strict_support_min_mm": float(support_df.loc[support_df["strict_support_gate_pass"], "amp_xy_mm"].min()) if support_df["strict_support_gate_pass"].any() else None,
        "strict_support_max_mm": float(support_df.loc[support_df["strict_support_gate_pass"], "amp_xy_mm"].max()) if support_df["strict_support_gate_pass"].any() else None,
        "relaxed_support_min_mm": float(support_df.loc[support_df["relaxed_support_gate_pass"], "amp_xy_mm"].min()) if support_df["relaxed_support_gate_pass"].any() else None,
        "relaxed_support_max_mm": float(support_df.loc[support_df["relaxed_support_gate_pass"], "amp_xy_mm"].max()) if support_df["relaxed_support_gate_pass"].any() else None,
    }
    write_json(out / "split_report.json", report)
    return report


def _base_task(
    args: argparse.Namespace,
    *,
    task_id: str,
    mode: str,
    config: ModelConfig,
    seed: int,
    result_path: Path,
    package_path: Path | None,
    prediction_path: Path | None,
    angle_stride: int,
    max_iter: int,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "mode": mode,
        "config": asdict(config),
        "seed": int(seed),
        "dataset": str(args.tube_dataset),
        "robot_config": str(args.robot_config),
        "result_path": str(result_path),
        "package_path": str(package_path) if package_path is not None else None,
        "prediction_path": str(prediction_path) if prediction_path is not None else None,
        "angle_stride": int(angle_stride),
        "max_iter": int(max_iter),
        "batch_size": 256,
    }


def _flatten_task_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "task_id": result["task_id"],
                "mode": result["mode"],
                "seed": result["seed"],
                "config_id": result["config_id"],
                **result["config"],
                **result["metrics"],
                "package_path": result.get("package_path"),
                "prediction_path": result.get("prediction_path"),
            }
        )
    return pd.DataFrame(rows)


def _rank_screen_runs(flat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config_id, part in flat.groupby("config_id", sort=False):
        first = part.iloc[0]
        rows.append(
            {
                "config_id": config_id,
                "architecture": first["architecture"],
                "hidden_layers": first["hidden_layers"],
                "feature_set": first["feature_set"],
                "activation": first["activation"],
                "alpha": float(first["alpha"]),
                "seed_count": int(len(part)),
                "all_seed_screen_gate_pass": bool(part["screen_gate_pass"].astype(bool).all()),
                "passed_seed_count": int(part["screen_gate_pass"].astype(bool).sum()),
                "val_ee_p95_median_mm": float(part["ee_p95_mm"].median()),
                "val_beta_p95_median_deg": float(part["beta_p95_deg"].median()),
                "val_axis_p95_median_mm": float(part["axiserr_max_p95_abs_mm"].median()),
                "fit_s_median": float(part["fit_s"].median()),
                "parameter_width_sum": int(sum(first["hidden_layers"])),
            }
        )
    ranked = pd.DataFrame(rows)
    return ranked.sort_values(
        ["all_seed_screen_gate_pass", "val_ee_p95_median_mm", "val_beta_p95_median_deg", "val_axis_p95_median_mm", "parameter_width_sum", "fit_s_median"],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)


def phase_screen(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "02_model_screen"
    result_dir = out / "worker_results"
    task_dir = out / "worker_tasks"
    out.mkdir(parents=True, exist_ok=True)
    settings = preset_settings(args.preset)
    seeds = parse_int_list(args.seeds)[: int(settings["screen_seed_count"])]
    configs = model_configs(activation="relu")
    limit = int(settings["screen_config_limit"])
    if limit > 0:
        configs = configs[:limit]
    tasks = []
    for config in configs:
        for seed in seeds:
            task_id = f"screen_{config.config_id}_s{seed}"
            tasks.append(
                _base_task(
                    args,
                    task_id=task_id,
                    mode="screen",
                    config=config,
                    seed=seed,
                    result_path=result_dir / f"{task_id}.json",
                    package_path=None,
                    prediction_path=None,
                    angle_stride=int(settings["angle_stride"]),
                    max_iter=int(settings["max_iter"]),
                )
            )
    results = run_tasks(tasks, task_dir=task_dir, workers=int(args.workers), skip_existing=bool(args.skip_existing))
    flat = _flatten_task_results(results)
    ranked = _rank_screen_runs(flat)
    fallback_triggered = bool(args.preset == "formal" and not ranked["all_seed_screen_gate_pass"].any())
    if fallback_triggered:
        top_pairs = [(str(row["architecture"]), str(row["feature_set"])) for _idx, row in ranked.head(2).iterrows()]
        fallback_configs = model_configs(activation="tanh", alphas=(1.0e-6, 1.0e-4), limited=top_pairs)
        fallback_tasks = []
        for config in fallback_configs:
            for seed in seeds:
                task_id = f"screen_{config.config_id}_s{seed}"
                fallback_tasks.append(
                    _base_task(
                        args,
                        task_id=task_id,
                        mode="screen",
                        config=config,
                        seed=seed,
                        result_path=result_dir / f"{task_id}.json",
                        package_path=None,
                        prediction_path=None,
                        angle_stride=int(settings["angle_stride"]),
                        max_iter=int(settings["max_iter"]),
                    )
                )
        fallback_results = run_tasks(fallback_tasks, task_dir=task_dir, workers=int(args.workers), skip_existing=bool(args.skip_existing))
        results.extend(fallback_results)
        flat = _flatten_task_results(results)
        ranked = _rank_screen_runs(flat)
    flat.to_csv(out / "screen_metrics_all_seeds.csv", index=False)
    ranked.to_csv(out / "screen_config_ranking.csv", index=False)
    selected_id = str(ranked.iloc[0]["config_id"])
    selected_result = next(result for result in results if result["config_id"] == selected_id)
    selected_config = dict(selected_result["config"])
    report = {
        "selected_config_id": selected_id,
        "selected_config": selected_config,
        "selected_all_seed_screen_gate_pass": bool(ranked.iloc[0]["all_seed_screen_gate_pass"]),
        "fallback_triggered": fallback_triggered,
        "screen_seed_count": len(seeds),
        "screen_run_count": int(len(flat)),
        "ranking_path": str(out / "screen_config_ranking.csv"),
    }
    write_json(out / "selection_report.json", report)
    return report


def phase_train(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "03_final_models"
    out.mkdir(parents=True, exist_ok=True)
    selection_path = Path(args.out_dir) / "02_model_screen" / "selection_report.json"
    if not selection_path.exists():
        raise RuntimeError("model screen selection report is missing")
    selection = read_json(selection_path)
    config = config_from_dict(dict(selection["selected_config"]))
    settings = preset_settings(args.preset)
    seeds = parse_int_list(args.seeds)[: int(settings["final_seed_count"])]
    tasks = []
    for seed in seeds:
        task_id = f"final_{config.config_id}_s{seed}"
        tasks.append(
            _base_task(
                args,
                task_id=task_id,
                mode="final",
                config=config,
                seed=seed,
                result_path=out / "worker_results" / f"{task_id}.json",
                package_path=out / "model_checkpoints" / f"seed_{seed}" / "model.joblib",
                prediction_path=out / "centerline_predictions" / f"seed_{seed}.parquet",
                angle_stride=int(settings["angle_stride"]),
                max_iter=int(settings["max_iter"]),
            )
        )
    results = run_tasks(tasks, task_dir=out / "worker_tasks", workers=int(args.workers), skip_existing=bool(args.skip_existing))
    flat = _flatten_task_results(results)
    flat.to_csv(out / "e75_centerline_metrics_all_seeds.csv", index=False)
    aggregate = aggregate_seed_gate(flat, required_fraction=0.8)

    sector_task_id = f"sector_{config.config_id}_s{seeds[0]}"
    sector_result = run_tasks(
        [
            _base_task(
                args,
                task_id=sector_task_id,
                mode="sector",
                config=config,
                seed=seeds[0],
                result_path=out / "sector_holdout" / f"{sector_task_id}.json",
                package_path=None,
                prediction_path=out / "sector_holdout" / "predictions.parquet",
                angle_stride=min(int(settings["angle_stride"]), 10),
                max_iter=int(settings["max_iter"]),
            )
        ],
        task_dir=out / "worker_tasks",
        workers=1,
        skip_existing=bool(args.skip_existing),
    )[0]

    diagnostic_result = None
    if not bool(aggregate["stable_gate_pass"]):
        diagnostic_id = f"full_diagnostic_{config.config_id}_s{seeds[0]}"
        diagnostic_result = run_tasks(
            [
                _base_task(
                    args,
                    task_id=diagnostic_id,
                    mode="full_diagnostic",
                    config=config,
                    seed=seeds[0],
                    result_path=out / "full_data_diagnostic" / f"{diagnostic_id}.json",
                    package_path=out / "full_data_diagnostic" / "model.joblib",
                    prediction_path=out / "full_data_diagnostic" / "centerline_predictions.parquet",
                    angle_stride=int(settings["angle_stride"]),
                    max_iter=int(settings["max_iter"]),
                )
            ],
            task_dir=out / "worker_tasks",
            workers=1,
            skip_existing=bool(args.skip_existing),
        )[0]
    report = {
        "selected_config_id": config.config_id,
        "selected_config": asdict(config),
        "centerline_holdout": True,
        "final_seed_gate": aggregate,
        "formal_radius_sweep_allowed": bool(aggregate["stable_gate_pass"]),
        "sector_holdout": sector_result["metrics"],
        "full_data_diagnostic": diagnostic_result,
        "metrics_path": str(out / "e75_centerline_metrics_all_seeds.csv"),
    }
    write_json(out / "final_training_report.json", report)
    return report


def _radius_sweep_rows(
    *,
    packages: list[tuple[int, dict[str, Any]]],
    radii: np.ndarray,
    metadata: dict[str, Any],
    training_xyz: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support_rows = []
    seed_rows = []
    for amp in radii:
        target, angle = generate_exact_ellipse(metadata, amp_xy_mm=float(amp), n_points=360)
        support = compute_support_metrics(training_xyz, target, radius_mm=15.0)
        strict_support = bool(support["nn_p95_mm"] <= 5.0 and support["nn_max_mm"] <= 8.0 and support["tube_count_p10"] >= 32.0)
        relaxed_support = bool(support["nn_p95_mm"] <= 8.0 and support["tube_count_p10"] >= 16.0)
        support_rows.append(
            {
                "amp_xy_mm": float(amp),
                "amp_z_mm": 1.5 * float(amp),
                **support,
                "strict_support_gate_pass": strict_support,
                "relaxed_support_gate_pass": relaxed_support,
            }
        )
        for seed, package in packages:
            beta_pred = predict_beta(package, target)
            metrics, _achieved, _theta = evaluate_beta_prediction(
                beta_pred=beta_pred,
                target_xyz=target,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
            )
            model_gate = centerline_model_gate(metrics, require_beta_error=False)
            seed_rows.append(
                {
                    "amp_xy_mm": float(amp),
                    "amp_z_mm": 1.5 * float(amp),
                    "seed": int(seed),
                    **support,
                    "strict_support_gate_pass": strict_support,
                    "relaxed_support_gate_pass": relaxed_support,
                    **metrics,
                    "model_gate_pass": bool(model_gate),
                }
            )
    return pd.DataFrame(seed_rows), pd.DataFrame(support_rows)


def _aggregate_radius_rows(seed_rows: pd.DataFrame, support_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for amp, part in seed_rows.groupby("amp_xy_mm", sort=True):
        support = support_rows[np.isclose(support_rows["amp_xy_mm"].to_numpy(dtype=float), float(amp))].iloc[0]
        seed_gate = aggregate_seed_gate(part, required_fraction=0.8)
        row: dict[str, Any] = {
            "amp_xy_mm": float(amp),
            "amp_z_mm": 1.5 * float(amp),
            **seed_gate,
            "strict_support_gate_pass": bool(support["strict_support_gate_pass"]),
            "relaxed_support_gate_pass": bool(support["relaxed_support_gate_pass"]),
        }
        for col in [
            "nn_p95_mm",
            "nn_max_mm",
            "tube_count_p10",
            "tube_count_min",
            "ee_mean_mm",
            "ee_p95_mm",
            "ee_max_mm",
            "axiserr_max_p95_abs_mm",
            "delta_beta_p95_deg",
            "delta_beta_max_deg",
            "delta2_beta_p95_deg",
            "seam_beta_rms_deg",
        ]:
            source = support_rows[np.isclose(support_rows["amp_xy_mm"].to_numpy(dtype=float), float(amp))][col] if col in support_rows.columns else part[col]
            if col in support_rows.columns:
                row[col] = float(source.iloc[0])
            else:
                row[f"{col}_median"] = float(part[col].median())
                row[f"{col}_p95_across_seeds"] = float(np.percentile(part[col].to_numpy(dtype=float), 95))
        row["strict_gate_pass"] = bool(seed_gate["stable_gate_pass"] and row["strict_support_gate_pass"])
        row["relaxed_gate_pass"] = bool(seed_gate["stable_gate_pass"] and row["relaxed_support_gate_pass"])
        row["model_only_gate_pass"] = bool(seed_gate["stable_gate_pass"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("amp_xy_mm").reset_index(drop=True)


def _representative_seed(seed_rows: pd.DataFrame, *, radius_mm: float) -> int:
    part = seed_rows[np.isclose(seed_rows["amp_xy_mm"].to_numpy(dtype=float), float(radius_mm))].copy()
    median = float(part["ee_p95_mm"].median())
    part["distance_to_median"] = np.abs(part["ee_p95_mm"].to_numpy(dtype=float) - median)
    return int(part.sort_values(["distance_to_median", "seed"]).iloc[0]["seed"])


def phase_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "04_radius_sweep"
    out.mkdir(parents=True, exist_ok=True)
    training_report = read_json(Path(args.out_dir) / "03_final_models" / "final_training_report.json")
    if not bool(training_report["formal_radius_sweep_allowed"]):
        report = {"sweep_executed": False, "reason": "e75_centerline_seed_gate_failed", "stable_radii": {"strict_supported_rmax_mm": None, "relaxed_supported_rmax_mm": None, "model_only_rmax_mm": None}}
        write_json(out / "stable_radius_report.json", report)
        return report
    df = pd.read_parquet(args.tube_dataset)
    metadata = tube_metadata(df)
    training = df[~df["is_centerline"].astype(bool)].copy()
    cfg = load_config(str(args.robot_config))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    metrics_df = pd.read_csv(Path(args.out_dir) / "03_final_models" / "e75_centerline_metrics_all_seeds.csv")
    packages = []
    for seed in metrics_df["seed"].astype(int).tolist():
        package_path = Path(args.out_dir) / "03_final_models" / "model_checkpoints" / f"seed_{seed}" / "model.joblib"
        packages.append((int(seed), load(package_path)))
    radii = _radius_values(args, preset=args.preset)
    seed_rows, support_rows = _radius_sweep_rows(
        packages=packages,
        radii=radii,
        metadata=metadata,
        training_xyz=training[TARGET_XYZ_COLS].to_numpy(dtype=float),
        lengths_m=inputs.lengths_m,
        p_end_local_m=inputs.p_end_local_m,
        theta_sign=theta_sign,
    )
    summary = _aggregate_radius_rows(seed_rows, support_rows)
    seed_rows.to_csv(out / "radius_sweep_all_seeds.csv", index=False)
    support_rows.to_csv(out / "radius_support.csv", index=False)
    summary.to_csv(out / "radius_sweep_summary.csv", index=False)
    stable = select_stable_radii(summary, anchor_mm=75.0)
    strict_limiting_factor = classify_strict_limit(summary, strict_rmax_mm=stable["strict_supported_rmax_mm"])
    chosen_radius = stable["strict_supported_rmax_mm"] or stable["relaxed_supported_rmax_mm"] or stable["model_only_rmax_mm"]
    representative_seed = _representative_seed(seed_rows, radius_mm=float(chosen_radius)) if chosen_radius is not None else None
    per_angle_path = None
    if chosen_radius is not None and representative_seed is not None:
        package = dict(packages)[int(representative_seed)]
        target, angle = generate_exact_ellipse(metadata, amp_xy_mm=float(chosen_radius), n_points=360)
        beta_pred = predict_beta(package, target)
        metrics, achieved, _theta = evaluate_beta_prediction(
            beta_pred=beta_pred,
            target_xyz=target,
            lengths_m=inputs.lengths_m,
            p_end_local_m=inputs.p_end_local_m,
            theta_sign=theta_sign,
        )
        per_angle = pd.DataFrame({"angle_idx": np.arange(360), "angle_rad": angle, "amp_xy_mm": float(chosen_radius), "seed": int(representative_seed)})
        for i, axis in enumerate("xyz"):
            per_angle[f"target_{axis}_m"] = target[:, i]
            per_angle[f"achieved_{axis}_m"] = achieved[:, i]
            per_angle[f"{axis}_err_mm"] = (achieved[:, i] - target[:, i]) * 1000.0
        per_angle["ee_err_mm"] = np.linalg.norm(achieved - target, axis=1) * 1000.0
        for i, col in enumerate(BETA_COLS):
            per_angle[f"pred_{col}"] = beta_pred[:, i]
        per_angle_path = out / "per_angle_error.parquet"
        per_angle.to_parquet(per_angle_path, index=False, compression="zstd")
        representative_metrics = metrics
    else:
        representative_metrics = None
    report = {
        "sweep_executed": True,
        "candidate_id": metadata["candidate_id"],
        "center_m": [metadata["center_x_m"], metadata["center_y_m"], metadata["center_z_m"]],
        "phase_y_deg": math.degrees(float(metadata["phase_y_rad"])),
        "phase_z_deg": math.degrees(float(metadata["phase_z_rad"])),
        "stable_radii": stable,
        "strict_limiting_factor": strict_limiting_factor,
        "strict_supported_amp_z_mm": 1.5 * float(stable["strict_supported_rmax_mm"]) if stable["strict_supported_rmax_mm"] is not None else None,
        "relaxed_supported_amp_z_mm": 1.5 * float(stable["relaxed_supported_rmax_mm"]) if stable["relaxed_supported_rmax_mm"] is not None else None,
        "model_only_amp_z_mm": 1.5 * float(stable["model_only_rmax_mm"]) if stable["model_only_rmax_mm"] is not None else None,
        "representative_radius_mm": chosen_radius,
        "representative_seed": representative_seed,
        "representative_metrics": representative_metrics,
        "per_angle_path": str(per_angle_path) if per_angle_path is not None else None,
    }
    write_json(out / "stable_radius_report.json", report)
    return report


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _equal_3d_limits(ax: Any, xyz: np.ndarray) -> None:
    values = np.asarray(xyz, dtype=float).reshape(-1, 3)
    lo = values.min(axis=0)
    hi = values.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float(np.max(hi - lo) * 0.54), 1.0e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def phase_visualize(args: argparse.Namespace) -> dict[str, Any]:
    sweep_dir = Path(args.out_dir) / "04_radius_sweep"
    out = Path(args.out_dir) / "05_visualization"
    out.mkdir(parents=True, exist_ok=True)
    report = read_json(sweep_dir / "stable_radius_report.json")
    summary_path = sweep_dir / "radius_sweep_summary.csv"
    generated: list[str] = []
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(summary["amp_xy_mm"], summary["ee_p95_mm_median"], label="median EE p95", color="#1f4e79")
        ax.plot(summary["amp_xy_mm"], summary["ee_p95_mm_p95_across_seeds"], label="seed p95 of EE p95", color="#c4511c")
        ax.axhline(5.0, color="#8b0000", ls="--", lw=1.0, label="EE gate 5 mm")
        ax.set_xlabel("amp_xy (mm)")
        ax.set_ylabel("EE p95 (mm)")
        ax.grid(alpha=0.25)
        ax.legend()
        path = out / "radius_vs_ee_p95.png"
        _savefig(fig, path)
        generated.append(str(path))

        fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
        ax1.plot(summary["amp_xy_mm"], summary["nn_p95_mm"], color="#1f4e79", label="NN p95")
        ax1.plot(summary["amp_xy_mm"], summary["nn_max_mm"], color="#417505", label="NN max")
        ax1.axhline(5.0, color="#1f4e79", ls="--", lw=0.9)
        ax1.axhline(8.0, color="#417505", ls="--", lw=0.9)
        ax1.set_xlabel("amp_xy (mm)")
        ax1.set_ylabel("distance (mm)")
        ax2 = ax1.twinx()
        ax2.plot(summary["amp_xy_mm"], summary["tube_count_p10"], color="#c4511c", label="tube count p10")
        ax2.axhline(32.0, color="#c4511c", ls="--", lw=0.9)
        ax2.set_ylabel("15 mm tube count p10")
        lines = ax1.get_lines()[:2] + ax2.get_lines()[:1]
        ax1.legend(lines, [line.get_label() for line in lines], loc="best")
        ax1.grid(alpha=0.2)
        path = out / "radius_vs_support.png"
        _savefig(fig, path)
        generated.append(str(path))

    seed_path = sweep_dir / "radius_sweep_all_seeds.csv"
    if seed_path.exists():
        seeds = pd.read_csv(seed_path)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for seed, part in seeds.groupby("seed"):
            ax.plot(part["amp_xy_mm"], part["ee_p95_mm"], lw=1.0, alpha=0.8, label=str(seed))
        ax.axhline(5.0, color="#222", ls="--", lw=0.9)
        ax.set_xlabel("amp_xy (mm)")
        ax.set_ylabel("EE p95 (mm)")
        ax.grid(alpha=0.2)
        ax.legend(title="seed", ncol=2, fontsize=8)
        path = out / "radius_seed_stability.png"
        _savefig(fig, path)
        generated.append(str(path))

    per_angle_path = sweep_dir / "per_angle_error.parquet"
    if per_angle_path.exists():
        points = pd.read_parquet(per_angle_path)
        target = points[[f"target_{axis}_m" for axis in "xyz"]].to_numpy(dtype=float)
        achieved = points[[f"achieved_{axis}_m" for axis in "xyz"]].to_numpy(dtype=float)
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(target[:, 0], target[:, 1], target[:, 2], color="#1f4e79", lw=2.0, label="target")
        ax.plot(achieved[:, 0], achieved[:, 1], achieved[:, 2], color="#c4511c", lw=1.6, label="MLP + FK")
        _equal_3d_limits(ax, np.vstack([target, achieved]))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=24, azim=-42)
        ax.legend()
        amp = float(points["amp_xy_mm"].iloc[0])
        ax.set_title(f"Stable true ellipse: a={amp:.2f} mm")
        path = out / "best_stable_ellipse_3d.png"
        _savefig(fig, path)
        generated.append(str(path))

        angle_deg = np.rad2deg(points["angle_rad"].to_numpy(dtype=float))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for axis in "xyz":
            ax.plot(angle_deg, points[f"{axis}_err_mm"], lw=1.2, label=f"{axis} error")
        ax.axhline(0.0, color="#222", lw=0.8)
        ax.set_xlabel("ellipse phase (deg)")
        ax.set_ylabel("axis error (mm)")
        ax.grid(alpha=0.2)
        ax.legend()
        path = out / "best_stable_ellipse_axis_errors.png"
        _savefig(fig, path)
        generated.append(str(path))

        beta = points[[f"pred_{col}" for col in BETA_COLS]].to_numpy(dtype=float) * 180.0 / math.pi
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for i in range(6):
            ax1.plot(angle_deg, beta[:, i], lw=1.0, label=f"beta{i + 1}")
        ax1.set_ylabel("predicted beta (deg)")
        ax1.legend(ncol=3, fontsize=8)
        ax1.grid(alpha=0.2)
        delta2 = _beta_rms_deg(np.roll(np.deg2rad(beta), -1, axis=0) - 2.0 * np.deg2rad(beta) + np.roll(np.deg2rad(beta), 1, axis=0))
        ax2.plot(angle_deg, delta2, color="#c4511c", lw=1.1)
        ax2.set_xlabel("ellipse phase (deg)")
        ax2.set_ylabel("cyclic delta2 beta RMS (deg)")
        ax2.grid(alpha=0.2)
        path = out / "beta_trajectory_and_smoothness.png"
        _savefig(fig, path)
        generated.append(str(path))
    payload = {"generated": generated, "stable_radius_report": report}
    write_json(out / "visualization_report.json", payload)
    return payload


def phase_summary(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "06_summary"
    out.mkdir(parents=True, exist_ok=True)
    audit = read_json(Path(args.out_dir) / "00_input_audit" / "audit_report.json")
    split = read_json(Path(args.out_dir) / "01_split_and_support" / "split_report.json")
    selection = read_json(Path(args.out_dir) / "02_model_screen" / "selection_report.json")
    training = read_json(Path(args.out_dir) / "03_final_models" / "final_training_report.json")
    radius = read_json(Path(args.out_dir) / "04_radius_sweep" / "stable_radius_report.json")
    stable = radius.get("stable_radii", {})
    strict = stable.get("strict_supported_rmax_mm")
    relaxed = stable.get("relaxed_supported_rmax_mm")
    model_only = stable.get("model_only_rmax_mm")
    limiting_factor = radius.get("strict_limiting_factor", "unknown")

    def fmt_mm(value: Any) -> str:
        return "not qualified" if value is None else f"{float(value):g}mm"

    metrics_path = Path(args.out_dir) / "03_final_models" / "e75_centerline_metrics_all_seeds.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    lines = [
        "# True Ellipse xyz -> beta6 Model Training V4 实验总结",
        "",
        "本实验使用 V3 选定 canonical branch 的 `±5mm` 法向 tube，训练 `xyz -> beta6 -> theta30 -> FK`。E75 中心线未进入训练集。",
        "",
        "## 数据与划分",
        "",
        f"- 输入数据审计：`{audit['audit_gate_pass']}`；rows=`{audit['rows']}`，angles=`{audit['unique_angles']}`，offset curves=`{audit['unique_offsets']}`。",
        f"- 筛选 train/validation/test rows：`{split['train_rows_screen']}/{split['validation_rows']}/{split['test_centerline_rows']}`。",
        f"- 最终训练 rows：`{split['final_training_rows']}`；中心线泄漏：`{split['centerline_overlap_with_train']}`。",
        f"- 轨迹相位保持为 py=`{math.degrees(float(audit['metadata']['phase_y_rad'])):.1f}deg`、pz=`{math.degrees(float(audit['metadata']['phase_z_rad'])):.1f}deg`。",
        "",
        "## 模型筛选",
        "",
        f"- 最优配置：`{selection['selected_config_id']}`。",
        f"- 第一轮未通过后触发 tanh/regularization fallback：`{selection['fallback_triggered']}`。",
        f"- 最终 E75 4/5 seed gate：`{training['final_seed_gate']['stable_gate_pass']}`（{training['final_seed_gate']['passed_seed_count']}/{training['final_seed_gate']['total_seed_count']}）。",
    ]
    if not metrics.empty:
        lines.extend(
            [
                f"- E75 EE p95 median：`{metrics['ee_p95_mm'].median():.6g}mm`；beta p95 median：`{metrics['beta_p95_deg'].median():.6g}deg`。",
                f"- E75 axis p95 median：`{metrics['axiserr_max_p95_abs_mm'].median():.6g}mm`；seed pass fraction：`{training['final_seed_gate']['seed_pass_fraction']:.3f}`。",
            ]
        )
    lines.extend(
        [
            f"- 330–360deg angle-sector holdout EE p95：`{float(training['sector_holdout']['ee_p95_mm']):.6g}mm`；diagnostic gate：`{training['sector_holdout']['model_gate_pass']}`。",
            "",
            "## 最大稳定半径",
            "",
            f"- 正式 strict support-backed amp_xy：`{fmt_mm(strict)}`；amp_z：`{fmt_mm(None if strict is None else 1.5 * float(strict))}`。",
            f"- relaxed support-backed amp_xy：`{fmt_mm(relaxed)}`。",
            f"- model-only 外推 amp_xy：`{fmt_mm(model_only)}`，该值不作为可靠数据支撑结论。",
            f"- 代表 seed：`{radius.get('representative_seed')}`，按五 seed 中位表现选择。",
            f"- 正式最大半径的首要限制因素：`{limiting_factor}`。",
            "",
            "## 结论边界",
            "",
            "- 正式半径同时受实际 8640 行训练池 support、4/5 seed 跟踪 gate、周期平滑和 beta bounds 约束。",
            "- 本结果只适用于当前中心、py=120deg、pz=30deg 的局部 canonical tube，不证明全工作空间存在全局单值逆映射。",
            "- 本阶段未训练张力、LGBM 或 direct-theta 模型，也未对模型输出执行 IK refinement。",
        ]
    )
    text = "\n".join(lines) + "\n"
    summary_path = out / "experiment_summary.md"
    summary_path.write_text(text, encoding="utf-8")
    docs_path = REPO_ROOT / "docs" / "TrueEllipseBeta6ModelV4实验记录.md" if str(args.preset) == "formal" else None
    if docs_path is not None:
        docs_path.write_text(text, encoding="utf-8")
    final = {
        "audit_gate_pass": audit["audit_gate_pass"],
        "selected_config_id": selection["selected_config_id"],
        "e75_seed_gate": training["final_seed_gate"],
        "stable_radii": stable,
        "summary_path": str(summary_path),
        "docs_path": str(docs_path) if docs_path is not None else None,
    }
    write_json(out / "final_report.json", final)
    return final


def run(args: argparse.Namespace) -> dict[str, Any]:
    phases = parse_phases(args.phases)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    started = time.perf_counter()
    phase_functions = {
        "audit": phase_audit,
        "split": phase_split,
        "screen": phase_screen,
        "train": phase_train,
        "sweep": phase_sweep,
        "visualize": phase_visualize,
        "summary": phase_summary,
    }
    for phase in phases:
        results[phase] = phase_functions[phase](args)
    payload = {
        "mode": "true_ellipse_beta6_training_v4",
        "preset": args.preset,
        "phases": phases,
        "elapsed_s": float(time.perf_counter() - started),
        "results": results,
    }
    write_json(Path(args.out_dir) / "run_report.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train xyz->beta6 models on the V3 canonical true-ellipse tube and find the stable radius.")
    ap.add_argument("--tube-dataset", type=Path, default=DEFAULT_TUBE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--preset", choices=["smoke", "pilot", "formal"], default="formal")
    ap.add_argument("--phases", default="all")
    ap.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    ap.add_argument("--radius-min-mm", type=float, default=50.0)
    ap.add_argument("--radius-max-mm", type=float, default=100.0)
    ap.add_argument("--radius-step-mm", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--worker-task", type=Path, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_task is not None:
        result = run_training_worker(read_json(args.worker_task))
        print(json.dumps({"task_id": result["task_id"], "ok": True}, ensure_ascii=False))
        return 0
    payload = run(args)
    final = payload.get("results", {}).get("summary", {})
    print(json.dumps({"out_dir": str(args.out_dir), "elapsed_s": payload["elapsed_s"], "stable_radii": final.get("stable_radii")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
