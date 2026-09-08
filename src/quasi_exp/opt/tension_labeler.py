from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from quasi_exp.opt.pso import solve_tensions_pso
from quasi_exp.opt.segmented_tension import solve_tensions_segmented
from quasi_exp.opt.tension_canonical import canonicalize_tension


@dataclass(frozen=True)
class TensionLabelResult:
    ok: bool
    T_base_12: np.ndarray
    meta: dict[str, Any]


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean(np.square(arr))))


def _method_from_cfg(model) -> str:
    cfg = getattr(model, "cfg", {}) or {}
    return str(cfg.get("tension_labeler", {}).get("method", "pso_canonical")).strip().lower()


def _case_flag_meta(cache) -> dict[str, Any]:
    return {"case_flag_12": np.asarray(cache.case_flag_12, dtype=int).tolist()}


def _solve_pso_canonical(model, cache, pso_cfg: dict[str, Any], pso_seed: int, rms_thresh: float) -> TensionLabelResult:
    pso_t0 = time.perf_counter()
    res = solve_tensions_pso(model, cache, pso_cfg=pso_cfg, rng_seed=pso_seed)
    pso_elapsed_s = float(time.perf_counter() - pso_t0)
    pso_rms_rnorm = float(np.sqrt(res.mean_rnorm2))
    T_base_12 = np.asarray(res.T_base_12, dtype=float)

    canonical_cfg = getattr(model, "cfg", {}).get("canonical_tension", {}) if hasattr(model, "cfg") else {}
    canonical_meta: dict[str, Any] = {
        "canonical_enabled": bool(canonical_cfg.get("enabled", False)),
        "canonical_adopted": False,
    }
    if bool(canonical_cfg.get("enabled", False)):
        c_res = canonicalize_tension(model, cache, [res.T_base_12], canonical_cfg)
        if bool(c_res.success) and float(c_res.rms_rnorm) < float(rms_thresh):
            T_base_12 = np.asarray(c_res.T_base_12, dtype=float)
            canonical_meta["canonical_adopted"] = True
        canonical_meta.update(
            {
                "canonical_success": bool(c_res.success),
                "canonical_method": c_res.method,
                "canonical_elapsed_s": float(c_res.elapsed_s),
                "canonical_nit": int(c_res.nit),
                "canonical_nfev": int(c_res.nfev),
                "canonical_objective": float(c_res.objective),
                "canonical_rms_rnorm": float(c_res.rms_rnorm),
                "canonical_mean_rnorm2": float(c_res.mean_rnorm2),
                "canonical_max_tension": float(c_res.max_tension),
                "canonical_source_index": int(c_res.source_index),
            }
        )
    else:
        canonical_meta.update({"canonical_success": False, "canonical_method": "disabled", "canonical_elapsed_s": 0.0})

    rnorm, _debug = model.residual_norm(cache, T_base_12)
    rms_rnorm = _rms(rnorm)
    mean_rnorm2 = float(np.mean(np.square(rnorm)))
    meta = {
        "tension_solver_method": "pso_canonical",
        "rms_rnorm": rms_rnorm,
        "mean_rnorm2": mean_rnorm2,
        "max_tension": float(np.max(T_base_12)),
        "best_cost": float(res.best_cost),
        "iters_used": int(res.iters_used),
        "evals": int(res.evals),
        "pso_seed": int(pso_seed),
        "pso_elapsed_s": pso_elapsed_s,
        "pso_rms_rnorm": pso_rms_rnorm,
        "pso_mean_rnorm2": float(res.mean_rnorm2),
        "pso_max_tension": float(res.max_tension),
        "pso_best_cost": float(res.best_cost),
        **_case_flag_meta(cache),
        **canonical_meta,
    }
    ok = bool(np.isfinite(T_base_12).all() and np.isfinite(rms_rnorm) and rms_rnorm < float(rms_thresh))
    return TensionLabelResult(ok=ok, T_base_12=T_base_12, meta=meta)


def _solve_segmented_canonical(model, cache, pso_seed: int, rms_thresh: float) -> TensionLabelResult:
    cfg = getattr(model, "cfg", {}) or {}
    solver_cfg = dict(cfg.get("segmented_tension", {}))
    if "feasible_rms_rnorm" not in solver_cfg:
        solver_cfg["feasible_rms_rnorm"] = float(rms_thresh)

    res = solve_tensions_segmented(model, cache, solver_cfg)
    T_base_12 = np.asarray(res.T_base_12, dtype=float)
    total_nfev = int(sum(int(v) for v in res.section_nfev.values()))
    meta: dict[str, Any] = {
        "tension_solver_method": "segmented_canonical",
        "rms_rnorm": float(res.rms_rnorm),
        "mean_rnorm2": float(res.mean_rnorm2),
        "max_tension": float(res.max_tension),
        "best_cost": float(res.mean_rnorm2),
        "iters_used": 0,
        "evals": total_nfev,
        "pso_seed": int(pso_seed),
        "pso_elapsed_s": 0.0,
        "canonical_enabled": True,
        "canonical_adopted": bool(res.success and res.rms_rnorm < float(rms_thresh)),
        "canonical_success": bool(res.success),
        "canonical_method": "segmented_canonical",
        "canonical_elapsed_s": float(res.elapsed_s),
        "canonical_nit": 0,
        "canonical_nfev": total_nfev,
        "canonical_objective": float(res.mean_rnorm2),
        "canonical_rms_rnorm": float(res.rms_rnorm),
        "canonical_mean_rnorm2": float(res.mean_rnorm2),
        "canonical_max_tension": float(res.max_tension),
        "canonical_source_index": 0,
        "segmented_success": bool(res.success),
        "segmented_elapsed_s": float(res.elapsed_s),
        "segmented_rms_rnorm": float(res.rms_rnorm),
        "segmented_mean_rnorm2": float(res.mean_rnorm2),
        "segmented_max_tension": float(res.max_tension),
        **_case_flag_meta(cache),
    }
    for name in ["third", "second", "first"]:
        meta[f"segmented_section_{name}_rms_rnorm"] = float(res.section_rms_rnorm.get(name, np.nan))
        meta[f"segmented_section_{name}_elapsed_s"] = float(res.section_elapsed_s.get(name, np.nan))
        meta[f"segmented_section_{name}_nfev"] = int(res.section_nfev.get(name, 0))

    ok = bool(np.isfinite(T_base_12).all() and bool(res.success) and float(res.rms_rnorm) < float(rms_thresh))
    return TensionLabelResult(ok=ok, T_base_12=T_base_12, meta=meta)


def solve_tension_label(
    *,
    model,
    cache,
    pso_cfg: dict[str, Any],
    pso_seed: int,
    rms_thresh: float,
) -> TensionLabelResult:
    method = _method_from_cfg(model)
    if method in {"pso_canonical", "pso", "legacy"}:
        return _solve_pso_canonical(model, cache, pso_cfg=pso_cfg, pso_seed=pso_seed, rms_thresh=rms_thresh)
    if method in {"segmented_canonical", "segmented"}:
        return _solve_segmented_canonical(model, cache, pso_seed=pso_seed, rms_thresh=rms_thresh)
    raise ValueError(f"Unsupported tension_labeler.method={method!r}")
