from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


STANDARD_SWEEP_AXIS_ORDER = tuple(f"beta{i}" for i in range(1, 7))


@dataclass(frozen=True)
class StandardSweepPlan:
    total_samples: int
    axis_order: tuple[str, ...]
    per_axis_positive_levels: dict[str, int]
    per_axis_rows: dict[str, int]
    per_axis_step_rad: dict[str, float]
    per_axis_step_deg: dict[str, float]
    axis_ranges_rad: dict[str, tuple[float, float]]


def beta_to_theta(beta6_rad: np.ndarray) -> np.ndarray:
    """
    论文 Eq.(50)：用 6 维 beta 生成 30 个关节角 theta。

    section1: i=1..10  odd->beta1 even->beta2
    section2: i=11..20 odd->beta3 even->beta4
    section3: i=21..30 odd->beta5 even->beta6
    """
    beta = np.asarray(beta6_rad, dtype=float).reshape(6)
    theta = np.zeros(30, dtype=float)
    for i in range(1, 31):
        section = (i - 1) // 10  # 0,1,2
        is_odd = (i % 2 == 1)
        if section == 0:
            theta[i - 1] = beta[0] if is_odd else beta[1]
        elif section == 1:
            theta[i - 1] = beta[2] if is_odd else beta[3]
        else:
            theta[i - 1] = beta[4] if is_odd else beta[5]
    return theta


def _resolve_standard_axis_order(axis_order: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if axis_order is None:
        return STANDARD_SWEEP_AXIS_ORDER
    order = tuple(str(axis).strip() for axis in axis_order)
    if order != STANDARD_SWEEP_AXIS_ORDER:
        raise ValueError(f"standard_sweep axis_order must be {list(STANDARD_SWEEP_AXIS_ORDER)}, got {list(order)}")
    return order


def _coerce_axis_ranges(
    beta_ranges_rad: dict[str, Any],
    axis_order: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for axis in axis_order:
        if axis not in beta_ranges_rad:
            raise ValueError(f"sampling.beta_ranges_rad missing {axis}")
        lo, hi = beta_ranges_rad[axis]
        out[axis] = (float(lo), float(hi))
    return out


def _axis_span_for_centered_sweep(lo: float, hi: float) -> float:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"standard_sweep beta range must be finite, got ({lo}, {hi})")
    if lo >= hi:
        raise ValueError(f"standard_sweep beta range must satisfy lo < hi, got ({lo}, {hi})")
    if not (lo < 0.0 < hi):
        raise ValueError(f"standard_sweep beta range must straddle 0, got ({lo}, {hi})")
    return float(min(abs(lo), abs(hi)))


def validate_generation_strategy(cfg: dict[str, Any]) -> str:
    sampling_cfg = cfg.get("sampling", {})
    strategy = str(sampling_cfg.get("strategy", "random")).strip().lower()
    dataset_mode = str(cfg.get("dataset", {}).get("mode", "forward")).strip().lower()
    if strategy == "standard_sweep" and dataset_mode != "forward":
        raise ValueError("sampling.strategy=standard_sweep only supports dataset.mode=forward")
    return strategy


def plan_standard_sweep_counts(
    beta_ranges_rad: dict[str, Any],
    num_samples: int,
    axis_order: list[str] | tuple[str, ...] | None = None,
) -> StandardSweepPlan:
    order = _resolve_standard_axis_order(axis_order)
    ranges = _coerce_axis_ranges(beta_ranges_rad, order)

    if num_samples < len(order):
        raise ValueError(f"standard_sweep requires at least {len(order)} samples, got {num_samples}")
    if num_samples % 2 != 0:
        raise ValueError(f"standard_sweep requires an even num_samples, got {num_samples}")

    positive_budget = (int(num_samples) - len(order)) // 2
    weights = np.asarray([_axis_span_for_centered_sweep(*ranges[axis]) for axis in order], dtype=float)
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0.0:
        raise ValueError("standard_sweep requires positive axis spans")

    raw = (weights / weights_sum) * float(positive_budget)
    base = np.floor(raw).astype(int)
    remainder = positive_budget - int(base.sum())
    if remainder > 0:
        fracs = raw - base
        ranked = sorted(range(len(order)), key=lambda idx: (-float(fracs[idx]), idx))
        for idx in ranked[:remainder]:
            base[idx] += 1

    per_axis_positive_levels = {axis: int(base[i]) for i, axis in enumerate(order)}
    per_axis_rows = {axis: int(1 + 2 * per_axis_positive_levels[axis]) for axis in order}
    per_axis_step_rad = {
        axis: float(weights[i] / per_axis_positive_levels[axis]) if per_axis_positive_levels[axis] > 0 else 0.0
        for i, axis in enumerate(order)
    }
    per_axis_step_deg = {
        axis: float(np.rad2deg(per_axis_step_rad[axis]))
        for axis in order
    }

    return StandardSweepPlan(
        total_samples=int(sum(per_axis_rows.values())),
        axis_order=order,
        per_axis_positive_levels=per_axis_positive_levels,
        per_axis_rows=per_axis_rows,
        per_axis_step_rad=per_axis_step_rad,
        per_axis_step_deg=per_axis_step_deg,
        axis_ranges_rad=ranges,
    )


def build_standard_sweep_tasks(
    beta_ranges_rad: dict[str, Any],
    num_samples: int,
    axis_order: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], StandardSweepPlan]:
    plan = plan_standard_sweep_counts(beta_ranges_rad=beta_ranges_rad, num_samples=num_samples, axis_order=axis_order)
    axis_to_idx = {axis: idx for idx, axis in enumerate(plan.axis_order)}

    tasks: list[dict[str, Any]] = []
    sample_id = 0
    for axis in plan.axis_order:
        axis_idx = axis_to_idx[axis]
        max_abs_angle = _axis_span_for_centered_sweep(*plan.axis_ranges_rad[axis])
        step_rad = plan.per_axis_step_rad[axis]
        beta_zero = np.zeros(6, dtype=float)
        tasks.append(
            {
                "sample_id": sample_id,
                "seed": sample_id + 1,
                "beta6_rad": beta_zero.tolist(),
                "scan_axis": axis,
                "scan_axis_idx": axis_idx + 1,
                "scan_sign": 0,
                "scan_level": 0,
                "scan_angle_rad": 0.0,
                "scan_angle_deg": 0.0,
            }
        )
        sample_id += 1

        for level in range(1, plan.per_axis_positive_levels[axis] + 1):
            signed_angles = []
            pos_angle = step_rad * float(level)
            if level == plan.per_axis_positive_levels[axis]:
                pos_angle = max_abs_angle
            signed_angles.append((1, pos_angle))
            signed_angles.append((-1, -pos_angle))
            for sign, angle in signed_angles:
                beta = np.zeros(6, dtype=float)
                beta[axis_idx] = float(angle)
                tasks.append(
                    {
                        "sample_id": sample_id,
                        "seed": sample_id + 1,
                        "beta6_rad": beta.tolist(),
                        "scan_axis": axis,
                        "scan_axis_idx": axis_idx + 1,
                        "scan_sign": int(sign),
                        "scan_level": int(level),
                        "scan_angle_rad": float(angle),
                        "scan_angle_deg": float(np.rad2deg(angle)),
                    }
                )
                sample_id += 1

    if len(tasks) != plan.total_samples:
        raise RuntimeError(f"standard_sweep built {len(tasks)} tasks but planned {plan.total_samples}")
    return tasks, plan


__all__ = [
    "STANDARD_SWEEP_AXIS_ORDER",
    "StandardSweepPlan",
    "beta_to_theta",
    "build_standard_sweep_tasks",
    "plan_standard_sweep_counts",
    "validate_generation_strategy",
]
