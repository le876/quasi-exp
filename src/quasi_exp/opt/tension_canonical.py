from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class CanonicalTensionResult:
    T_base_12: np.ndarray
    objective: float
    mean_rnorm2: float
    rms_rnorm: float
    max_tension: float
    success: bool
    method: str
    nit: int
    nfev: int
    elapsed_s: float
    message: str
    source_index: int


def _rms_rnorm(model, cache: Any, T_base_12: np.ndarray) -> tuple[float, float]:
    rnorm, _debug = model.residual_norm(cache, np.asarray(T_base_12, dtype=float).reshape(12))
    mean_r2 = float(np.mean(np.square(rnorm)))
    return float(np.sqrt(mean_r2)), mean_r2


def _as_t_ref(raw: Any, tmin: float, tmax: float) -> np.ndarray:
    if raw is None:
        value = 0.5 * (tmin + tmax)
        return np.full(12, value, dtype=float)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        return np.full(12, float(arr), dtype=float)
    return np.clip(arr.reshape(12), tmin, tmax).astype(float)


def _smooth_max_ratio(T: np.ndarray, tmax: float, temp: float) -> float:
    x = np.asarray(T, dtype=float).reshape(12) / max(tmax, 1.0)
    tau = max(float(temp), 1.0e-6)
    z = x / tau
    zmax = float(np.max(z))
    return float(tau * (zmax + np.log(np.sum(np.exp(z - zmax)))))


def _canonical_objective(
    T: np.ndarray,
    *,
    tmax: float,
    t_ref: np.ndarray,
    t_prev: np.ndarray | None,
    w_max: float,
    w_ref: float,
    w_prev: float,
    smoothmax_temp: float,
) -> float:
    T = np.asarray(T, dtype=float).reshape(12)
    tscale = max(float(tmax), 1.0)
    ref_term = float(np.mean(np.square((T - t_ref) / tscale)))
    max_term = _smooth_max_ratio(T, tmax=tscale, temp=smoothmax_temp)
    prev_term = 0.0
    if t_prev is not None:
        prev_term = float(np.mean(np.square((T - t_prev) / tscale)))
    return float(w_max * max_term + w_ref * ref_term + w_prev * prev_term)


def _unique_initials(initial_tensions: Iterable[np.ndarray], t_ref: np.ndarray, tmin: float, tmax: float) -> list[np.ndarray]:
    initials: list[np.ndarray] = []
    for raw in list(initial_tensions) + [t_ref]:
        arr = np.clip(np.asarray(raw, dtype=float).reshape(12), tmin, tmax)
        if not np.isfinite(arr).all():
            continue
        if any(np.allclose(arr, prev, rtol=0.0, atol=1.0e-9) for prev in initials):
            continue
        initials.append(arr.astype(float))
    if not initials:
        initials.append(t_ref.copy())
    return initials


def canonicalize_tension(
    model,
    cache: Any,
    initial_tensions: Iterable[np.ndarray],
    cfg: dict[str, Any] | None,
    *,
    T_prev: np.ndarray | None = None,
) -> CanonicalTensionResult:
    cfg = dict(cfg or {})
    t0 = time.perf_counter()
    tmin = float(getattr(model, "t_min", 0.0))
    tmax = float(getattr(model, "t_max", 2000.0))
    t_ref = _as_t_ref(cfg.get("t_ref_n", cfg.get("reference_n")), tmin=tmin, tmax=tmax)
    initials = _unique_initials(initial_tensions, t_ref=t_ref, tmin=tmin, tmax=tmax)

    fallback = initials[0].copy()
    fallback_rms, fallback_mean_r2 = _rms_rnorm(model, cache, fallback)
    threshold = float(cfg.get("rms_rnorm_threshold", cfg.get("residual_rms_threshold", 6.0e-2)))
    if (not np.isfinite(threshold)) or threshold <= 0.0:
        threshold = 6.0e-2

    if not bool(cfg.get("enabled", True)):
        return CanonicalTensionResult(
            T_base_12=fallback,
            objective=float("nan"),
            mean_rnorm2=fallback_mean_r2,
            rms_rnorm=fallback_rms,
            max_tension=float(np.max(fallback)),
            success=True,
            method="disabled",
            nit=0,
            nfev=0,
            elapsed_s=float(time.perf_counter() - t0),
            message="canonical_tension disabled",
            source_index=0,
        )

    w_max = float(cfg.get("w_max", 1.0))
    w_ref = float(cfg.get("w_ref", 0.2))
    w_prev = float(cfg.get("w_prev", 0.0))
    smoothmax_temp = float(cfg.get("smoothmax_temp", 0.02))
    maxiter = int(cfg.get("maxiter", 30))
    ftol = float(cfg.get("ftol", 1.0e-6))
    accept_slack = float(cfg.get("accept_rms_slack", 1.0e-7))
    t_prev = None if T_prev is None else np.asarray(T_prev, dtype=float).reshape(12)
    method_cfg = str(cfg.get("method", "slsqp_ref")).strip().lower()

    def obj(T: np.ndarray) -> float:
        return _canonical_objective(
            T,
            tmax=tmax,
            t_ref=t_ref,
            t_prev=t_prev,
            w_max=w_max,
            w_ref=w_ref,
            w_prev=w_prev,
            smoothmax_temp=smoothmax_temp,
        )

    if method_cfg in {"slsqp_penalty_ref", "penalty_ref", "slsqp_soft_ref"}:
        residual_violation_weight = float(cfg.get("residual_violation_weight", 1.0e4))
        residual_fit_weight = float(cfg.get("residual_fit_weight", 50.0))

        def penalty_obj(T: np.ndarray) -> float:
            rms, _mean_r2 = _rms_rnorm(model, cache, np.asarray(T, dtype=float).reshape(12))
            if not np.isfinite(rms):
                return float("inf")
            violation = max(0.0, rms - threshold)
            return float(
                residual_violation_weight * violation * violation
                + residual_fit_weight * rms * rms
                + obj(T)
            )

        opt = minimize(
            penalty_obj,
            x0=t_ref.copy(),
            method="SLSQP",
            bounds=[(tmin, tmax)] * 12,
            options={"maxiter": maxiter, "ftol": ftol, "disp": False},
        )
        T = np.clip(np.asarray(opt.x, dtype=float).reshape(12), tmin, tmax)
        rms, mean_r2 = _rms_rnorm(model, cache, T)
        feasible = bool(np.isfinite(rms) and rms <= threshold + accept_slack)
        if feasible:
            return CanonicalTensionResult(
                T_base_12=T,
                objective=float(obj(T)),
                mean_rnorm2=float(mean_r2),
                rms_rnorm=float(rms),
                max_tension=float(np.max(T)),
                success=True,
                method="slsqp_penalty_ref",
                nit=int(getattr(opt, "nit", 0)),
                nfev=int(getattr(opt, "nfev", 0)),
                elapsed_s=float(time.perf_counter() - t0),
                message=str(getattr(opt, "message", "")),
                source_index=len(initials),
            )

    def constraint(T: np.ndarray) -> float:
        rms, _mean_r2 = _rms_rnorm(model, cache, np.asarray(T, dtype=float).reshape(12))
        if not np.isfinite(rms):
            return -1.0e9
        return float(threshold - rms)

    best: CanonicalTensionResult | None = None
    total_nfev = 0
    total_nit = 0
    messages: list[str] = []
    bounds = [(tmin, tmax)] * 12
    for idx, x0 in enumerate(initials):
        opt = minimize(
            obj,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=({"type": "ineq", "fun": constraint},),
            options={"maxiter": maxiter, "ftol": ftol, "disp": False},
        )
        total_nfev += int(getattr(opt, "nfev", 0))
        total_nit += int(getattr(opt, "nit", 0))
        messages.append(str(getattr(opt, "message", "")))
        T = np.clip(np.asarray(opt.x, dtype=float).reshape(12), tmin, tmax)
        rms, mean_r2 = _rms_rnorm(model, cache, T)
        feasible = bool(np.isfinite(rms) and rms <= threshold + accept_slack)
        if not feasible:
            continue
        value = obj(T)
        candidate = CanonicalTensionResult(
            T_base_12=T,
            objective=float(value),
            mean_rnorm2=float(mean_r2),
            rms_rnorm=float(rms),
            max_tension=float(np.max(T)),
            success=True,
            method="slsqp",
            nit=total_nit,
            nfev=total_nfev,
            elapsed_s=float(time.perf_counter() - t0),
            message="; ".join(m for m in messages if m),
            source_index=idx,
        )
        if best is None or candidate.objective < best.objective:
            best = candidate

    if best is not None:
        return CanonicalTensionResult(
            T_base_12=best.T_base_12,
            objective=best.objective,
            mean_rnorm2=best.mean_rnorm2,
            rms_rnorm=best.rms_rnorm,
            max_tension=best.max_tension,
            success=True,
            method=best.method,
            nit=best.nit,
            nfev=best.nfev,
            elapsed_s=float(time.perf_counter() - t0),
            message=best.message,
            source_index=best.source_index,
        )

    return CanonicalTensionResult(
        T_base_12=fallback,
        objective=float(obj(fallback)) if np.isfinite(fallback).all() else float("inf"),
        mean_rnorm2=fallback_mean_r2,
        rms_rnorm=fallback_rms,
        max_tension=float(np.max(fallback)),
        success=False,
        method="fallback_pso",
        nit=total_nit,
        nfev=total_nfev,
        elapsed_s=float(time.perf_counter() - t0),
        message="; ".join(m for m in messages if m) or "no feasible canonical solution",
        source_index=0,
    )
