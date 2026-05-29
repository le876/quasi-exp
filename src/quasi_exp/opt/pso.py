from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .losses import normalized_pseudo_huber


@dataclass(frozen=True)
class PsoResult:
    T_base_12: np.ndarray
    best_cost: float
    mean_rnorm2: float
    max_tension: float
    iters_used: int
    evals: int


def _tension_objective_cost(
    T_base: np.ndarray,
    mean_r2: float,
    tmax: float,
    pso_cfg: dict[str, Any],
) -> float:
    if tmax <= 0:
        return float("inf")
    if not np.isfinite(mean_r2):
        return float("inf")

    objective = str(pso_cfg.get("objective", "weighted_sum")).strip().lower()
    T_base = np.asarray(T_base, dtype=float).reshape(-1)
    if not np.isfinite(T_base).all():
        return float("inf")

    rms_rnorm = float(np.sqrt(float(mean_r2)))
    max_ratio = float(np.max(T_base) / tmax)

    if objective == "paper_constraint":
        feasible_rms = float(pso_cfg.get("paper_feasible_rms_rnorm", 0.06))
        if (not np.isfinite(feasible_rms)) or feasible_rms <= 0.0:
            feasible_rms = 0.06
        residual_ratio = rms_rnorm / feasible_rms
        if rms_rnorm <= feasible_rms:
            return float(max_ratio + 1.0e-3 * residual_ratio)
        return float(1.0e6 + residual_ratio**2 + max_ratio)

    if objective != "weighted_sum":
        raise ValueError(f"Unsupported pso.objective={objective!r}")

    w_resid = float(pso_cfg["w_resid"])
    lambda_max = float(pso_cfg["lambda_max"])
    w_tension_mean = float(pso_cfg.get("w_tension_mean", 0.0))
    w_tension_soft_cap = float(pso_cfg.get("w_tension_soft_cap", 0.0))
    tension_soft_cap_ratio = float(pso_cfg.get("tension_soft_cap_ratio", 1.0))
    max_tension_power = float(pso_cfg.get("max_tension_power", 1.0))
    normalized_objectives = bool(pso_cfg.get("normalized_objectives", False))

    t_norm = T_base / tmax
    max_term = float(max_ratio**max_tension_power)
    mean_term = float(np.mean(np.square(t_norm)))
    cap = float(np.clip(tension_soft_cap_ratio, 0.0, 1.0))
    excess = np.maximum(0.0, t_norm - cap)
    soft_cap_term = float(np.mean(np.square(excess)))

    resid_term = float(mean_r2)
    if normalized_objectives:
        resid_term = float(
            normalized_pseudo_huber(
                rms_rnorm,
                scale=float(pso_cfg.get("residual_cost_scale", 0.045)),
                delta=float(pso_cfg.get("residual_cost_delta", 0.8)),
            )
        )

    return float(
        w_resid * resid_term
        + lambda_max * max_term
        + w_tension_mean * mean_term
        + w_tension_soft_cap * soft_cap_term
    )


def solve_tensions_pso(
    model,
    cache,
    pso_cfg: dict[str, Any],
    rng_seed: int,
) -> PsoResult:
    backend = str(pso_cfg.get("backend", "numpy")).strip().lower()
    if backend in {"tensorflow", "tf", "tensorflow_gpu"}:
        from .pso_tf import solve_tensions_pso_tf

        return solve_tensions_pso_tf(model=model, cache=cache, pso_cfg=pso_cfg, rng_seed=rng_seed)

    n_particles = int(pso_cfg["n_particles"])
    iters = int(pso_cfg["iters"])
    inertia = float(pso_cfg["inertia"])
    c1 = float(pso_cfg["c1"])
    c2 = float(pso_cfg["c2"])
    early = float(pso_cfg["early_stop_mean_rnorm2"])

    tmin = float(model.t_min)
    tmax = float(model.t_max)

    rng = np.random.default_rng(int(rng_seed))

    # 初始化：均匀采样
    X = rng.uniform(tmin, tmax, size=(n_particles, 12)).astype(float)
    V = rng.normal(0.0, (tmax - tmin) * 0.05, size=(n_particles, 12)).astype(float)

    pbest = X.copy()
    pbest_cost = np.full(n_particles, np.inf, dtype=float)
    gbest = X[0].copy()
    gbest_cost = np.inf
    gbest_mean_r2 = np.inf
    evals = 0

    def cost_fn(T_base: np.ndarray) -> tuple[float, float]:
        nonlocal evals
        evals += 1
        rnorm, _debug = model.residual_norm(cache, T_base)
        mean_r2 = float(np.mean(np.square(rnorm)))
        J = _tension_objective_cost(T_base, mean_r2=mean_r2, tmax=tmax, pso_cfg=pso_cfg)
        if not np.isfinite(J):
            return float("inf"), mean_r2
        return float(J), mean_r2

    # 初始评估
    for p in range(n_particles):
        X[p] = np.clip(X[p], tmin, tmax)
        J, mean_r2 = cost_fn(X[p])
        pbest_cost[p] = J
        if J < gbest_cost:
            gbest_cost = J
            gbest = X[p].copy()
            gbest_mean_r2 = mean_r2

    if gbest_mean_r2 < early:
        return PsoResult(
            T_base_12=gbest,
            best_cost=float(gbest_cost),
            mean_rnorm2=float(gbest_mean_r2),
            max_tension=float(np.max(gbest)),
            iters_used=0,
            evals=evals,
        )

    for it in range(1, iters + 1):
        r1 = rng.random(size=(n_particles, 12))
        r2 = rng.random(size=(n_particles, 12))
        V = inertia * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
        X = X + V
        X = np.clip(X, tmin, tmax)

        for p in range(n_particles):
            J, mean_r2 = cost_fn(X[p])
            if J < pbest_cost[p]:
                pbest_cost[p] = J
                pbest[p] = X[p].copy()
            if J < gbest_cost:
                gbest_cost = J
                gbest = X[p].copy()
                gbest_mean_r2 = mean_r2

        if gbest_mean_r2 < early:
            return PsoResult(
                T_base_12=gbest,
                best_cost=float(gbest_cost),
                mean_rnorm2=float(gbest_mean_r2),
                max_tension=float(np.max(gbest)),
                iters_used=it,
                evals=evals,
            )

    return PsoResult(
        T_base_12=gbest,
        best_cost=float(gbest_cost),
        mean_rnorm2=float(gbest_mean_r2),
        max_tension=float(np.max(gbest)),
        iters_used=iters,
        evals=evals,
    )
