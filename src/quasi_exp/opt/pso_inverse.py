from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quasi_exp.model.kinematics import forward_kinematics
from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.opt.losses import normalized_pseudo_huber, theta_delta_rms_deg
from quasi_exp.opt.pso import solve_tensions_pso


@dataclass(frozen=True)
class InverseJointPsoResult:
    beta6_rad: np.ndarray
    theta_rad: np.ndarray
    T_base_12: np.ndarray
    p_xyz_m: np.ndarray
    target_xyz_m: np.ndarray
    xyz_err_m: float
    best_cost: float
    mean_rnorm2: float
    rms_rnorm: float
    max_tension: float
    iters_used: int
    evals: int
    case_flag_12: np.ndarray


def _theta_delta_mse(beta_a: np.ndarray, beta_b: np.ndarray) -> float:
    theta_a = beta_to_theta(np.asarray(beta_a, dtype=float).reshape(6))
    theta_b = beta_to_theta(np.asarray(beta_b, dtype=float).reshape(6))
    return float(np.mean(np.square(theta_a - theta_b)))


def _use_normalized_objectives(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("normalized_objectives", False))


def _normalized_xyz_penalty(xyz_err_m: float, cfg: dict[str, Any]) -> float:
    return float(
        normalized_pseudo_huber(
            xyz_err_m,
            scale=float(cfg.get("xyz_cost_scale_m", 0.015)),
            delta=float(cfg.get("xyz_cost_delta", 0.6)),
        )
    )


def _normalized_theta_delta_penalty(beta_a: np.ndarray, beta_b: np.ndarray, cfg: dict[str, Any]) -> float:
    return float(
        normalized_pseudo_huber(
            theta_delta_rms_deg(beta_a, beta_b),
            scale=float(cfg.get("cont_cost_scale_deg", 6.0)),
            delta=float(cfg.get("cont_cost_delta", 0.4)),
        )
    )


def _resolve_theta123_beta_indices(inverse_pso_cfg: dict[str, Any]) -> tuple[int, int, int]:
    raw = inverse_pso_cfg.get("theta123_beta_indices", [0, 2, 4])
    try:
        idx = [int(v) for v in raw]
    except Exception:  # noqa: BLE001
        idx = [0, 2, 4]
    if len(idx) != 3:
        idx = [0, 2, 4]
    if len(set(idx)) != 3:
        idx = [0, 2, 4]
    if any(v < 0 or v > 5 for v in idx):
        idx = [0, 2, 4]
    return int(idx[0]), int(idx[1]), int(idx[2])


def _theta1_priority_xyz_err_threshold_m(inverse_pso_cfg: dict[str, Any]) -> float:
    thr = float(inverse_pso_cfg.get("theta1_priority_xyz_err_m", 0.02))
    if (not np.isfinite(thr)) or (thr <= 0.0):
        return 0.02
    return thr


def _refine_theta23_with_fixed_theta1(
    model,
    inputs,
    target_xyz_m: np.ndarray,
    beta_seed: np.ndarray,
    beta_ranges_rad: dict[str, Any],
    inverse_pso_cfg: dict[str, Any],
    tension_pso_cfg: dict[str, Any],
    rng_seed: int,
    continuity_ref_beta: np.ndarray | None = None,
) -> InverseJointPsoResult:
    idx1, idx2, idx3 = _resolve_theta123_beta_indices(inverse_pso_cfg)
    beta0 = np.asarray(beta_seed, dtype=float).reshape(6).copy()
    target = np.asarray(target_xyz_m, dtype=float).reshape(3)
    bounds = _beta_bounds(beta_ranges_rad)
    beta_scale = np.maximum(np.maximum(np.abs(bounds[:, 0]), np.abs(bounds[:, 1])), 1e-9)

    beta0[idx1] = float(np.clip(beta0[idx1], bounds[idx1, 0], bounds[idx1, 1]))
    beta0[idx2] = float(np.clip(beta0[idx2], bounds[idx2, 0], bounds[idx2, 1]))
    beta0[idx3] = float(np.clip(beta0[idx3], bounds[idx3, 0], bounds[idx3, 1]))

    n_particles = max(8, int(inverse_pso_cfg.get("theta23_refine_particles", 16)))
    iters = max(10, int(inverse_pso_cfg.get("theta23_refine_iters", 36)))
    inertia = float(inverse_pso_cfg.get("inertia", 0.72))
    c1 = float(inverse_pso_cfg.get("c1", 1.49))
    c2 = float(inverse_pso_cfg.get("c2", 1.49))
    w_xyz = float(inverse_pso_cfg.get("w_xyz", 3.0e5))
    w_beta_l2 = float(inverse_pso_cfg.get("w_beta_l2", 0.0))
    w_refine_cont = float(
        inverse_pso_cfg.get(
            "theta23_refine_w_continuity",
            inverse_pso_cfg.get("select_w_delta_theta", 0.0),
        )
    )

    cont_ref = None
    if continuity_ref_beta is not None:
        cont_ref = np.asarray(continuity_ref_beta, dtype=float).reshape(6)

    lo = np.array([bounds[idx2, 0], bounds[idx3, 0]], dtype=float)
    hi = np.array([bounds[idx2, 1], bounds[idx3, 1]], dtype=float)
    scale = np.maximum(hi - lo, 1e-9)
    rng = np.random.default_rng(int(rng_seed) + 5003001)

    X = rng.uniform(lo[None, :], hi[None, :], size=(n_particles, 2))
    X[0] = np.array([beta0[idx2], beta0[idx3]], dtype=float)
    V = rng.normal(0.0, scale[None, :] * 0.08, size=(n_particles, 2))

    pbest = X.copy()
    pbest_cost = np.full(n_particles, np.inf, dtype=float)
    gbest = X[0].copy()
    gbest_cost = np.inf
    evals = 0

    def cost_fn(x2: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evals
        evals += 1
        beta = beta0.copy()
        beta[idx2] = float(np.clip(x2[0], lo[0], hi[0]))
        beta[idx3] = float(np.clip(x2[1], lo[1], hi[1]))
        theta_raw = beta_to_theta(beta)
        p_xyz, _ = forward_kinematics(
            theta_raw,
            inputs.lengths_m,
            inputs.p_end_local_m,
            theta_sign=model.theta_sign,
        )
        xyz_err = float(np.linalg.norm(p_xyz - target))
        beta_term = float(np.mean(np.square(beta / beta_scale)))
        if _use_normalized_objectives(inverse_pso_cfg):
            base_cost = w_xyz * _normalized_xyz_penalty(xyz_err, inverse_pso_cfg) + w_beta_l2 * beta_term
        else:
            base_cost = w_xyz * (xyz_err**2) + w_beta_l2 * beta_term
        J = base_cost
        if (cont_ref is not None) and (w_refine_cont > 0.0):
            if _use_normalized_objectives(inverse_pso_cfg):
                J += w_refine_cont * _normalized_theta_delta_penalty(beta, cont_ref, inverse_pso_cfg)
            else:
                delta = beta[[idx2, idx3]] - cont_ref[[idx2, idx3]]
                delta_scale = np.maximum(beta_scale[[idx2, idx3]], 1e-9)
                J += w_refine_cont * float(np.mean(np.square(delta / delta_scale)))
        if not np.isfinite(J):
            return float("inf"), beta
        return float(J), beta

    for p in range(n_particles):
        J, _ = cost_fn(X[p])
        pbest_cost[p] = J
        if J < gbest_cost:
            gbest_cost = J
            gbest = X[p].copy()

    for _ in range(iters):
        r1 = rng.random(size=(n_particles, 2))
        r2 = rng.random(size=(n_particles, 2))
        V = inertia * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
        X = np.clip(X + V, lo[None, :], hi[None, :])
        for p in range(n_particles):
            J, _ = cost_fn(X[p])
            if J < pbest_cost[p]:
                pbest_cost[p] = J
                pbest[p] = X[p].copy()
            if J < gbest_cost:
                gbest_cost = J
                gbest = X[p].copy()

    _, beta_best = cost_fn(gbest)
    return _finalize_result(
        model=model,
        inputs=inputs,
        target_xyz_m=target,
        best_beta=beta_best,
        best_cost=float(gbest_cost),
        iters_used=int(iters),
        evals=int(evals),
        tension_pso_cfg=tension_pso_cfg,
        rng_seed=int(rng_seed) + 7005001,
    )


def _tension_penalty_score(tensions: np.ndarray, tension_pso_cfg: dict[str, Any], tmax: float) -> float:
    tensions = np.asarray(tensions, dtype=float).reshape(-1)
    if (not np.isfinite(tensions).all()) or (tmax <= 0.0):
        return float("inf")

    lambda_max = float(tension_pso_cfg.get("lambda_max", 0.0))
    max_tension_power = float(tension_pso_cfg.get("max_tension_power", 1.0))
    w_tension_mean = float(tension_pso_cfg.get("w_tension_mean", 0.0))
    w_tension_soft_cap = float(tension_pso_cfg.get("w_tension_soft_cap", 0.0))
    cap_ratio = float(np.clip(tension_pso_cfg.get("tension_soft_cap_ratio", 1.0), 0.0, 1.0))

    t_norm = tensions / float(tmax)
    max_term = float(np.max(t_norm) ** max_tension_power)
    mean_term = float(np.mean(np.square(t_norm)))
    excess = np.maximum(0.0, t_norm - cap_ratio)
    soft_cap_term = float(np.mean(np.square(excess)))

    return float(
        lambda_max * max_term
        + w_tension_mean * mean_term
        + w_tension_soft_cap * soft_cap_term
    )


def _resolve_canonical_penalty_cfg(
    inverse_pso_cfg: dict[str, Any],
    tension_pso_cfg: dict[str, Any],
) -> dict[str, Any]:
    penalty_keys = (
        "lambda_max",
        "max_tension_power",
        "w_tension_mean",
        "w_tension_soft_cap",
        "tension_soft_cap_ratio",
    )
    resolved: dict[str, Any] = {}
    for key in penalty_keys:
        if key in inverse_pso_cfg:
            resolved[key] = inverse_pso_cfg[key]
        elif key in tension_pso_cfg:
            resolved[key] = tension_pso_cfg[key]
    return resolved


def _canonicalize_beta_from_candidates(
    model,
    inputs,
    target_xyz_m: np.ndarray,
    candidate_betas: np.ndarray,
    inverse_pso_cfg: dict[str, Any],
    tension_pso_cfg: dict[str, Any],
    rng_seed: int,
    continuity_ref_beta: np.ndarray | None = None,
) -> np.ndarray:
    mode = str(inverse_pso_cfg.get("canonical_mode", "none")).strip().lower()
    if mode not in {"paper_minmax_tension"}:
        return np.asarray(candidate_betas, dtype=float).reshape(-1, 6)[0].copy()

    candidates = np.asarray(candidate_betas, dtype=float).reshape(-1, 6)
    if candidates.shape[0] < 1:
        raise ValueError("candidate_betas is empty")

    rounded = np.round(candidates, decimals=10)
    _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
    candidates = candidates[np.sort(uniq_idx)]

    xyz_errs = np.zeros(candidates.shape[0], dtype=float)
    for i in range(candidates.shape[0]):
        theta_raw = beta_to_theta(candidates[i])
        p_xyz, _ = forward_kinematics(
            theta_raw,
            inputs.lengths_m,
            inputs.p_end_local_m,
            theta_sign=model.theta_sign,
        )
        xyz_errs[i] = float(np.linalg.norm(p_xyz - target_xyz_m))

    best_xyz_err = float(np.min(xyz_errs))
    xyz_tol_m = float(inverse_pso_cfg.get("canonical_xyz_tol_m", 0.003))
    top_k = max(1, int(inverse_pso_cfg.get("canonical_top_k", 3)))
    xyz_tie_weight = float(inverse_pso_cfg.get("canonical_w_xyz_tie", 0.0))
    delta_theta_weight = float(
        inverse_pso_cfg.get("canonical_w_delta_theta", inverse_pso_cfg.get("select_w_delta_theta", 0.0))
    )
    penalty_cfg = _resolve_canonical_penalty_cfg(inverse_pso_cfg, tension_pso_cfg)

    near_idx = np.where(xyz_errs <= (best_xyz_err + xyz_tol_m))[0]
    if near_idx.size == 0:
        near_idx = np.argsort(xyz_errs)[:top_k]
    else:
        near_idx = near_idx[np.argsort(xyz_errs[near_idx])][:top_k]

    best_idx = int(np.argmin(xyz_errs))
    best_score = float("inf")
    theta_ref = None
    if continuity_ref_beta is not None:
        theta_ref = beta_to_theta(np.asarray(continuity_ref_beta, dtype=float).reshape(6))

    for rank, idx in enumerate(near_idx.tolist()):
        beta = candidates[int(idx)]
        theta_raw = beta_to_theta(beta)
        cache = model.build_cache(theta_raw)
        tension_seed = int(rng_seed) + 2000003 + int(idx) + rank * 997
        t_res = solve_tensions_pso(model, cache, pso_cfg=tension_pso_cfg, rng_seed=tension_seed)
        tension_score = _tension_penalty_score(
            np.asarray(t_res.T_base_12, dtype=float),
            tension_pso_cfg=penalty_cfg,
            tmax=float(model.t_max),
        )
        delta_theta_term = 0.0
        xyz_term = float(xyz_errs[int(idx)] ** 2)
        if _use_normalized_objectives(inverse_pso_cfg):
            xyz_term = _normalized_xyz_penalty(float(xyz_errs[int(idx)]), inverse_pso_cfg)
        if (theta_ref is not None) and (delta_theta_weight > 0.0):
            if _use_normalized_objectives(inverse_pso_cfg):
                delta_theta_term = _normalized_theta_delta_penalty(beta, continuity_ref_beta, inverse_pso_cfg)
            else:
                theta_cur = beta_to_theta(beta)
                delta_theta_term = float(np.mean(np.square(theta_cur - theta_ref)))
        total_score = float(
            tension_score
            + xyz_tie_weight * xyz_term
            + delta_theta_weight * delta_theta_term
        )
        if (total_score < best_score) or (
            np.isclose(total_score, best_score) and (xyz_errs[int(idx)] < xyz_errs[best_idx])
        ):
            best_score = total_score
            best_idx = int(idx)

    return candidates[best_idx].copy()


def _beta_bounds(beta_ranges_rad: dict[str, Any]) -> np.ndarray:
    bounds = np.zeros((6, 2), dtype=float)
    for i in range(6):
        key = f"beta{i+1}"
        if key not in beta_ranges_rad:
            raise ValueError(f"sampling.beta_ranges_rad missing {key}")
        lo, hi = beta_ranges_rad[key]
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise ValueError(f"Invalid beta range for {key}: {beta_ranges_rad[key]}")
        bounds[i, 0] = lo
        bounds[i, 1] = hi
    return bounds


def _select_best_restart_result(
    results: list[InverseJointPsoResult],
    model,
    inverse_pso_cfg: dict[str, Any],
    tension_pso_cfg: dict[str, Any],
    continuity_ref_beta: np.ndarray | None,
) -> InverseJointPsoResult:
    if len(results) == 1:
        return results[0]

    w_sel_xyz = float(inverse_pso_cfg.get("select_w_xyz", inverse_pso_cfg.get("w_xyz", 1.0)))
    w_sel_delta = float(inverse_pso_cfg.get("select_w_delta_theta", 0.0))
    penalty_cfg = _resolve_canonical_penalty_cfg(inverse_pso_cfg, tension_pso_cfg)

    best_idx = 0
    best_score = float("inf")
    for idx, res in enumerate(results):
        t_score = _tension_penalty_score(
            np.asarray(res.T_base_12, dtype=float),
            tension_pso_cfg=penalty_cfg,
            tmax=float(model.t_max),
        )
        if _use_normalized_objectives(inverse_pso_cfg):
            score = float(t_score + w_sel_xyz * _normalized_xyz_penalty(float(res.xyz_err_m), inverse_pso_cfg))
        else:
            score = float(t_score + w_sel_xyz * (float(res.xyz_err_m) ** 2))
        if (continuity_ref_beta is not None) and (w_sel_delta > 0.0):
            if _use_normalized_objectives(inverse_pso_cfg):
                score += float(w_sel_delta * _normalized_theta_delta_penalty(res.beta6_rad, continuity_ref_beta, inverse_pso_cfg))
            else:
                score += float(w_sel_delta * _theta_delta_mse(res.beta6_rad, continuity_ref_beta))
        if score < best_score:
            best_score = score
            best_idx = idx
    return results[best_idx]


def _solve_inverse_joint_pso_single(
    model,
    inputs,
    xyz_target_m: np.ndarray,
    beta_ranges_rad: dict[str, Any],
    inverse_pso_cfg: dict[str, Any],
    tension_pso_cfg: dict[str, Any],
    rng_seed: int,
    warm_start_beta: np.ndarray | None = None,
    continuity_ref_beta: np.ndarray | None = None,
) -> InverseJointPsoResult:
    backend = str(inverse_pso_cfg.get("backend", "numpy")).strip().lower()
    if backend in {"tensorflow", "tf", "tensorflow_gpu"}:
        from .pso_inverse_tf import solve_inverse_joint_pso_tf

        return solve_inverse_joint_pso_tf(
            model=model,
            inputs=inputs,
            xyz_target_m=xyz_target_m,
            beta_ranges_rad=beta_ranges_rad,
            inverse_pso_cfg=inverse_pso_cfg,
            tension_pso_cfg=tension_pso_cfg,
            rng_seed=rng_seed,
            warm_start_beta=warm_start_beta,
            continuity_ref_beta=continuity_ref_beta,
        )

    n_particles = int(inverse_pso_cfg["n_particles"])
    iters = int(inverse_pso_cfg["iters"])
    inertia = float(inverse_pso_cfg["inertia"])
    c1 = float(inverse_pso_cfg["c1"])
    c2 = float(inverse_pso_cfg["c2"])

    w_xyz = float(inverse_pso_cfg["w_xyz"])
    w_beta_l2 = float(inverse_pso_cfg.get("w_beta_l2", 0.0))
    early_xyz = float(inverse_pso_cfg.get("early_stop_xyz_err_m", 0.0))
    early_r2 = float(inverse_pso_cfg.get("early_stop_mean_rnorm2", 0.0))
    w_cont = float(inverse_pso_cfg.get("w_continuity", 0.0))
    cont_ratio_cap = float(np.clip(inverse_pso_cfg.get("continuity_max_ratio", 0.3), 0.0, 1.0))
    cont_relax_if_err_ratio = float(inverse_pso_cfg.get("continuity_relax_if_err_ratio", 1.2))
    cont_relax_scale = float(np.clip(inverse_pso_cfg.get("continuity_relax_scale", 0.3), 0.0, 1.0))
    warm_start_enabled = bool(inverse_pso_cfg.get("warm_start", True))
    warm_start_particles = max(1, int(inverse_pso_cfg.get("warm_start_particles", 6)))
    warm_start_sigma = max(0.0, float(inverse_pso_cfg.get("warm_start_sigma", 0.08)))

    beta_bounds = _beta_bounds(beta_ranges_rad)
    beta_lo = beta_bounds[:, 0]
    beta_hi = beta_bounds[:, 1]
    beta_scale = np.maximum(np.maximum(np.abs(beta_lo), np.abs(beta_hi)), 1e-9)

    target = np.asarray(xyz_target_m, dtype=float).reshape(3)
    rng = np.random.default_rng(int(rng_seed))

    dim = 6
    X = rng.uniform(beta_lo[None, :], beta_hi[None, :], size=(n_particles, dim)).astype(float)
    X[0] = np.zeros(6, dtype=float)

    V = np.zeros((n_particles, dim), dtype=float)
    beta_vel_scale = np.maximum(beta_hi - beta_lo, 1e-6) * 0.08
    V = rng.normal(0.0, beta_vel_scale[None, :], size=(n_particles, 6)).astype(float)

    continuity_ref = None
    if continuity_ref_beta is not None:
        continuity_ref = np.asarray(continuity_ref_beta, dtype=float).reshape(6)

    if warm_start_enabled and (warm_start_beta is not None) and np.isfinite(warm_start_beta).all():
        warm = np.clip(np.asarray(warm_start_beta, dtype=float).reshape(6), beta_lo, beta_hi)
        X[0] = warm
        warm_n = min(n_particles, warm_start_particles)
        if warm_n > 1:
            noise = rng.normal(0.0, beta_vel_scale[None, :] * warm_start_sigma, size=(warm_n - 1, 6))
            X[1:warm_n] = np.clip(warm[None, :] + noise, beta_lo[None, :], beta_hi[None, :])

    pbest = X.copy()
    pbest_cost = np.full(n_particles, np.inf, dtype=float)
    gbest = X[0].copy()
    gbest_cost = np.inf
    gbest_xyz_err = np.inf
    evals = 0

    def clamp_population() -> None:
        X[:, :6] = np.clip(X[:, :6], beta_lo[None, :], beta_hi[None, :])

    def cost_fn(x_beta: np.ndarray) -> tuple[float, float]:
        nonlocal evals
        evals += 1

        beta = x_beta[:6]
        theta_raw = beta_to_theta(beta)
        p_xyz, _ = forward_kinematics(
            theta_raw,
            inputs.lengths_m,
            inputs.p_end_local_m,
            theta_sign=model.theta_sign,
        )
        xyz_err = float(np.linalg.norm(p_xyz - target))

        beta_term = float(np.mean(np.square(beta / beta_scale)))
        if _use_normalized_objectives(inverse_pso_cfg):
            base_cost = w_xyz * _normalized_xyz_penalty(xyz_err, inverse_pso_cfg) + w_beta_l2 * beta_term
        else:
            base_cost = w_xyz * (xyz_err**2) + w_beta_l2 * beta_term
        J = base_cost
        if (continuity_ref is not None) and (w_cont > 0.0):
            if _use_normalized_objectives(inverse_pso_cfg):
                delta_term = _normalized_theta_delta_penalty(beta, continuity_ref, inverse_pso_cfg)
            else:
                delta_term = float(np.mean(np.square((beta - continuity_ref) / beta_scale)))
            eff_w_cont = w_cont
            if (early_xyz > 0.0) and (xyz_err > cont_relax_if_err_ratio * early_xyz):
                eff_w_cont *= cont_relax_scale
            cont_cost = eff_w_cont * delta_term
            if cont_ratio_cap > 0.0:
                cont_cost = min(cont_cost, cont_ratio_cap * max(base_cost, 1e-12))
            J = base_cost + cont_cost

        if not np.isfinite(J):
            return float("inf"), xyz_err
        return float(J), xyz_err

    clamp_population()
    for p in range(n_particles):
        J, xyz_err = cost_fn(X[p])
        pbest_cost[p] = J
        if J < gbest_cost:
            gbest_cost = J
            gbest = X[p].copy()
            gbest_xyz_err = xyz_err
    stop_iter = 0
    early_stop = bool(gbest_xyz_err <= early_xyz)

    if not early_stop:
        for it in range(1, iters + 1):
            r1 = rng.random(size=(n_particles, dim))
            r2 = rng.random(size=(n_particles, dim))
            V = inertia * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
            X = X + V
            clamp_population()

            for p in range(n_particles):
                J, xyz_err = cost_fn(X[p])
                if J < pbest_cost[p]:
                    pbest_cost[p] = J
                    pbest[p] = X[p].copy()
                if J < gbest_cost:
                    gbest_cost = J
                    gbest = X[p].copy()
                    gbest_xyz_err = xyz_err

            if gbest_xyz_err <= early_xyz:
                stop_iter = it
                early_stop = True
                break

    if not early_stop:
        stop_iter = iters

    mode = str(inverse_pso_cfg.get("canonical_mode", "none")).strip().lower()
    if mode in {"paper_minmax_tension"}:
        cand = np.vstack([gbest.reshape(1, 6), pbest])
        gbest = _canonicalize_beta_from_candidates(
            model=model,
            inputs=inputs,
            target_xyz_m=target,
            candidate_betas=cand,
            inverse_pso_cfg=inverse_pso_cfg,
            tension_pso_cfg=tension_pso_cfg,
            rng_seed=int(rng_seed),
            continuity_ref_beta=continuity_ref,
        )

    return _finalize_result(model, inputs, target, gbest, gbest_cost, stop_iter, evals, tension_pso_cfg, rng_seed)


def solve_inverse_joint_pso(
    model,
    inputs,
    xyz_target_m: np.ndarray,
    beta_ranges_rad: dict[str, Any],
    inverse_pso_cfg: dict[str, Any],
    tension_pso_cfg: dict[str, Any],
    rng_seed: int,
    warm_start_beta: np.ndarray | None = None,
    continuity_ref_beta: np.ndarray | None = None,
) -> InverseJointPsoResult:
    n_restarts = max(1, int(inverse_pso_cfg.get("n_restarts", 1)))
    restart_seed_stride = max(1, int(inverse_pso_cfg.get("restart_seed_stride", 1009)))
    warm_start_all_restarts = bool(inverse_pso_cfg.get("warm_start_all_restarts", False))

    results: list[InverseJointPsoResult] = []
    for ridx in range(n_restarts):
        seed = int(rng_seed) + ridx * restart_seed_stride
        warm_for_this = warm_start_beta if (ridx == 0 or warm_start_all_restarts) else None
        results.append(
            _solve_inverse_joint_pso_single(
                model=model,
                inputs=inputs,
                xyz_target_m=xyz_target_m,
                beta_ranges_rad=beta_ranges_rad,
                inverse_pso_cfg=inverse_pso_cfg,
                tension_pso_cfg=tension_pso_cfg,
                rng_seed=seed,
                warm_start_beta=warm_for_this,
                continuity_ref_beta=continuity_ref_beta,
            )
        )
    selected = _select_best_restart_result(
        results=results,
        model=model,
        inverse_pso_cfg=inverse_pso_cfg,
        tension_pso_cfg=tension_pso_cfg,
        continuity_ref_beta=continuity_ref_beta,
    )
    if not bool(inverse_pso_cfg.get("theta1_priority_enable", False)):
        return selected

    idx1, _, _ = _resolve_theta123_beta_indices(inverse_pso_cfg)
    theta1_min_result = min(results, key=lambda r: (float(r.beta6_rad[idx1]), float(r.xyz_err_m)))
    xyz_thr = _theta1_priority_xyz_err_threshold_m(inverse_pso_cfg)
    if float(theta1_min_result.xyz_err_m) <= xyz_thr:
        return theta1_min_result

    refined = _refine_theta23_with_fixed_theta1(
        model=model,
        inputs=inputs,
        target_xyz_m=xyz_target_m,
        beta_seed=theta1_min_result.beta6_rad,
        beta_ranges_rad=beta_ranges_rad,
        inverse_pso_cfg=inverse_pso_cfg,
        tension_pso_cfg=tension_pso_cfg,
        rng_seed=int(rng_seed),
        continuity_ref_beta=continuity_ref_beta,
    )
    if np.isfinite(refined.xyz_err_m):
        return refined
    return theta1_min_result


def _finalize_result(
    model,
    inputs,
    target_xyz_m: np.ndarray,
    best_beta: np.ndarray,
    best_cost: float,
    iters_used: int,
    evals: int,
    tension_pso_cfg: dict[str, Any],
    rng_seed: int,
) -> InverseJointPsoResult:
    beta = np.asarray(best_beta[:6], dtype=float).copy()
    theta_raw = beta_to_theta(beta)
    theta = theta_raw * float(model.theta_sign)
    cache = model.build_cache(theta_raw)
    tension_seed = int(rng_seed) + 1000003
    t_res = solve_tensions_pso(model, cache, pso_cfg=tension_pso_cfg, rng_seed=tension_seed)
    tensions = np.asarray(t_res.T_base_12, dtype=float).copy()
    rnorm, _ = model.residual_norm(cache, tensions)
    mean_r2 = float(np.mean(np.square(rnorm)))
    rms_rnorm = float(np.sqrt(mean_r2))
    p_xyz, _ = forward_kinematics(
        theta_raw,
        inputs.lengths_m,
        inputs.p_end_local_m,
        theta_sign=model.theta_sign,
    )
    xyz_err = float(np.linalg.norm(p_xyz - target_xyz_m))

    return InverseJointPsoResult(
        beta6_rad=beta,
        theta_rad=theta,
        T_base_12=tensions,
        p_xyz_m=np.asarray(p_xyz, dtype=float),
        target_xyz_m=np.asarray(target_xyz_m, dtype=float),
        xyz_err_m=xyz_err,
        best_cost=float(best_cost),
        mean_rnorm2=mean_r2,
        rms_rnorm=float(rms_rnorm),
        max_tension=float(np.max(tensions)),
        iters_used=int(iters_used + int(t_res.iters_used)),
        evals=int(evals + int(t_res.evals)),
        case_flag_12=cache.case_flag_12.astype(int).copy(),
    )
