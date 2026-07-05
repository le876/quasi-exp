from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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


@dataclass(frozen=True)
class MixedBetaPlan:
    total_samples: int
    component_counts: dict[str, int]
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


def effective_beta_from_theta(theta30_rad: np.ndarray) -> np.ndarray:
    """
    Recover the 6 independent section angles from a theta matrix.

    This is the canonical beta used for diagnostics and branch policy.  It is
    derived from the actual theta labels, so it avoids relying on metadata beta
    columns whose sign may differ from the stored theta convention.
    """
    theta = np.asarray(theta30_rad, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 30:
        raise ValueError("theta must have shape [N, 30]")
    cols = []
    for section in range(3):
        start = section * 10
        odd_idx = [start + i for i in range(0, 10, 2)]
        even_idx = [start + i for i in range(1, 10, 2)]
        cols.append(theta[:, odd_idx].mean(axis=1))
        cols.append(theta[:, even_idx].mean(axis=1))
    return np.stack(cols, axis=1)


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
    if strategy == "mixed_beta" and dataset_mode != "forward":
        raise ValueError("sampling.strategy=mixed_beta only supports dataset.mode=forward")
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


def _allocate_component_counts(num_samples: int, components: dict[str, Any]) -> dict[str, int]:
    if num_samples <= 0:
        raise ValueError("mixed_beta requires num_samples > 0")
    if not components:
        components = {
            "sobol_full": 0.4,
            "lhs_full": 0.2,
            "workspace_balanced": 0.2,
            "distal_biased": 0.2,
        }

    names = [str(k).strip() for k in components.keys()]
    weights = np.asarray([float(components[k]) for k in components.keys()], dtype=float)
    if any(not name for name in names):
        raise ValueError("mixed_beta component names must be non-empty")
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
        raise ValueError("mixed_beta component weights must be finite and positive")
    if np.any(weights < 0.0):
        raise ValueError("mixed_beta component weights must be non-negative")

    weights = weights / float(np.sum(weights))
    raw = weights * int(num_samples)
    base = np.floor(raw).astype(int)
    remainder = int(num_samples) - int(base.sum())
    if remainder > 0:
        ranked = sorted(range(len(names)), key=lambda idx: (-float(raw[idx] - base[idx]), idx))
        for idx in ranked[:remainder]:
            base[idx] += 1
    return {name: int(base[i]) for i, name in enumerate(names)}


def _scale_unit_to_beta(unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    U = np.asarray(unit, dtype=float).reshape(-1, 6)
    return bounds[:, 0][None, :] + np.clip(U, 0.0, 1.0) * (bounds[:, 1] - bounds[:, 0])[None, :]


def _sobol_unit(n: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 6), dtype=float)
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=6, scramble=True, seed=int(seed))
        m = int(np.ceil(np.log2(max(1, int(n)))))
        return np.asarray(sampler.random_base2(m), dtype=float)[: int(n)]
    except Exception:  # noqa: BLE001
        rng = np.random.default_rng(int(seed))
        return rng.random((int(n), 6))


def _lhs_unit(n: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 6), dtype=float)
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=6, optimization="random-cd", seed=int(seed))
        return np.asarray(sampler.random(int(n)), dtype=float)
    except Exception:  # noqa: BLE001
        rng = np.random.default_rng(int(seed))
        return rng.random((int(n), 6))


def _uniform_beta(n: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    U = rng.random((int(n), 6))
    return _scale_unit_to_beta(U, bounds)


def _distal_biased_beta(n: int, bounds: np.ndarray, cfg: dict[str, Any], seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = int(n)
    beta = np.zeros((n, 6), dtype=float)
    if n <= 0:
        return beta

    prox_max = float(cfg.get("proximal_abs_ratio_max", 0.45))
    distal_min = float(cfg.get("distal_abs_ratio_min", 0.55))
    prox_max = float(np.clip(prox_max, 0.0, 1.0))
    distal_min = float(np.clip(distal_min, 0.0, 1.0))
    if distal_min > 1.0:
        distal_min = 1.0

    spans = np.maximum(np.maximum(np.abs(bounds[:, 0]), np.abs(bounds[:, 1])), 1e-12)
    prox_mag = rng.uniform(0.0, prox_max, size=(n, 4))
    dist_mag = rng.uniform(distal_min, 1.0, size=(n, 2))
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=(n, 6))
    beta[:, :4] = signs[:, :4] * prox_mag * spans[:4][None, :]
    beta[:, 4:] = signs[:, 4:] * dist_mag * spans[4:][None, :]
    return np.clip(beta, bounds[:, 0][None, :], bounds[:, 1][None, :])


def _joint_group_norms(beta: np.ndarray, bounds: np.ndarray) -> tuple[float, float, float]:
    scale = np.maximum(np.maximum(np.abs(bounds[:, 0]), np.abs(bounds[:, 1])), 1e-12)
    b = np.asarray(beta, dtype=float).reshape(6)
    g1 = float(np.sqrt(np.mean(np.square(b[[0, 1]] / scale[[0, 1]]))))
    g2 = float(np.sqrt(np.mean(np.square(b[[2, 3]] / scale[[2, 3]]))))
    g3 = float(np.sqrt(np.mean(np.square(b[[4, 5]] / scale[[4, 5]]))))
    return g1, g2, g3


def _distal_preference_score(beta: np.ndarray, bounds: np.ndarray) -> float:
    g1, g2, g3 = _joint_group_norms(beta, bounds)
    return float(g1**2 + g2**2 - 1.25 * g3**2)


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    v = np.asarray(values, dtype=float).reshape(-1)
    n_bins = max(1, int(n_bins))
    if v.size == 0 or n_bins == 1:
        return np.zeros(v.size, dtype=int)
    qs = np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.searchsorted(qs, v, side="right").astype(int)


def _select_workspace_balanced(
    beta_pool: np.ndarray,
    xyz_pool: np.ndarray,
    n: int,
    cfg: dict[str, Any],
) -> np.ndarray:
    beta = np.asarray(beta_pool, dtype=float).reshape(-1, 6)
    xyz = np.asarray(xyz_pool, dtype=float).reshape(-1, 3)
    n = int(n)
    if beta.shape[0] <= n:
        return beta[:n].copy()

    r_bins = _quantile_bins(np.linalg.norm(xyz, axis=1), int(cfg.get("radius_bins", 5)))
    z_bins = _quantile_bins(xyz[:, 2], int(cfg.get("z_bins", 5)))
    angle = np.arctan2(xyz[:, 2], xyz[:, 1])
    angle_bins = np.floor(((angle + np.pi) / (2.0 * np.pi)) * max(1, int(cfg.get("angle_bins", 8)))).astype(int)
    angle_bins = np.clip(angle_bins, 0, max(1, int(cfg.get("angle_bins", 8))) - 1)

    buckets: dict[tuple[int, int, int], list[int]] = {}
    for idx, key in enumerate(zip(r_bins.tolist(), z_bins.tolist(), angle_bins.tolist())):
        buckets.setdefault(tuple(int(v) for v in key), []).append(int(idx))

    selected: list[int] = []
    bucket_keys = sorted(buckets.keys())
    cursor = 0
    while len(selected) < n and bucket_keys:
        key = bucket_keys[cursor % len(bucket_keys)]
        vals = buckets[key]
        if vals:
            selected.append(vals.pop(0))
        if not vals:
            bucket_keys.remove(key)
            if not bucket_keys:
                break
            cursor = cursor % len(bucket_keys)
        else:
            cursor += 1

    if len(selected) < n:
        used = set(selected)
        for idx in range(beta.shape[0]):
            if idx not in used:
                selected.append(int(idx))
                if len(selected) >= n:
                    break
    return beta[np.asarray(selected[:n], dtype=int)].copy()


def build_mixed_beta_tasks(
    beta_ranges_rad: dict[str, Any],
    num_samples: int,
    mixed_cfg: dict[str, Any] | None = None,
    rng_seed: int = 0,
    workspace_xyz_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], MixedBetaPlan]:
    """
    Build full-combination 6D beta sampling tasks.

    The mixed strategy preserves global coverage while adding a distal-biased
    component that uses the third joint more than the first two joints.
    """
    cfg = dict(mixed_cfg or {})
    order = STANDARD_SWEEP_AXIS_ORDER
    ranges = _coerce_axis_ranges(beta_ranges_rad, order)
    bounds = np.asarray([ranges[axis] for axis in order], dtype=float)
    counts = _allocate_component_counts(int(num_samples), dict(cfg.get("components", {})))

    component_betas: list[tuple[str, np.ndarray]] = []
    for offset, (name, count) in enumerate(counts.items()):
        if count <= 0:
            continue
        seed = int(rng_seed) + 7919 * (offset + 1)
        if name == "sobol_full":
            beta = _scale_unit_to_beta(_sobol_unit(count, seed), bounds)
        elif name == "lhs_full":
            beta = _scale_unit_to_beta(_lhs_unit(count, seed), bounds)
        elif name == "workspace_balanced":
            wb_cfg = dict(cfg.get("workspace_balanced", {}))
            multiplier = max(1, int(wb_cfg.get("candidate_multiplier", 8)))
            pool_n = max(count, int(count) * multiplier)
            beta_pool = _scale_unit_to_beta(_sobol_unit(pool_n, seed + 101), bounds)
            if workspace_xyz_fn is None:
                beta = beta_pool[:count]
            else:
                xyz_pool = workspace_xyz_fn(beta_pool)
                beta = _select_workspace_balanced(beta_pool, xyz_pool, count, wb_cfg)
        elif name == "distal_biased":
            beta = _distal_biased_beta(count, bounds, dict(cfg.get("distal_biased", {})), seed)
        else:
            beta = _uniform_beta(count, bounds, seed)
        component_betas.append((name, beta))

    tasks: list[dict[str, Any]] = []
    sample_id = 0
    for name, beta_rows in component_betas:
        for component_idx, beta in enumerate(np.asarray(beta_rows, dtype=float).reshape(-1, 6)):
            g1, g2, g3 = _joint_group_norms(beta, bounds)
            tasks.append(
                {
                    "sample_id": sample_id,
                    "seed": sample_id + 1,
                    "beta6_rad": beta.tolist(),
                    "source_component": name,
                    "source_component_idx": int(component_idx),
                    "beta_group1_norm": g1,
                    "beta_group2_norm": g2,
                    "beta_group3_norm": g3,
                    "distal_preference_score": _distal_preference_score(beta, bounds),
                }
            )
            sample_id += 1

    if len(tasks) != int(num_samples):
        raise RuntimeError(f"mixed_beta built {len(tasks)} tasks but expected {num_samples}")
    plan = MixedBetaPlan(total_samples=int(num_samples), component_counts=counts, axis_ranges_rad=ranges)
    return tasks, plan


__all__ = [
    "STANDARD_SWEEP_AXIS_ORDER",
    "MixedBetaPlan",
    "StandardSweepPlan",
    "beta_to_theta",
    "effective_beta_from_theta",
    "build_mixed_beta_tasks",
    "build_standard_sweep_tasks",
    "plan_standard_sweep_counts",
    "validate_generation_strategy",
]
