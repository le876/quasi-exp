from __future__ import annotations

from typing import Any

import numpy as np

from .pso import PsoResult


def _build_transmission_multipliers(
    theta_rad: np.ndarray,
    case_flag_12: np.ndarray,
    end_disk_by_j: dict[int, int],
    mu_cable: float,
    kD: int,
) -> np.ndarray:
    theta = np.asarray(theta_rad, dtype=float).reshape(kD)
    case_flag = np.asarray(case_flag_12, dtype=int).reshape(12)

    theta_ext = np.zeros(kD + 1, dtype=float)
    theta_ext[1:] = theta

    mult = np.zeros((kD + 1, 12), dtype=float)
    mult[0, :] = 1.0

    for j in range(1, 13):
        end_disk = int(end_disk_by_j[j])
        cflag = int(case_flag[j - 1])
        for i in range(0, min(end_disk, kD)):
            th_i = float(theta_ext[i])
            th_ip1 = float(theta_ext[i + 1])
            if cflag == 1:
                num = np.cos(th_i / 2.0) + mu_cable * np.sin(th_i / 2.0)
                den = np.cos(th_ip1 / 2.0) - mu_cable * np.sin(th_ip1 / 2.0)
            else:
                num = np.cos(th_i / 2.0) - mu_cable * np.sin(th_i / 2.0)
                den = np.cos(th_ip1 / 2.0) + mu_cable * np.sin(th_ip1 / 2.0)
            ratio = float(num / den)
            if np.isfinite(ratio):
                mult[i + 1, j - 1] = mult[i, j - 1] * ratio
            else:
                mult[i + 1, j - 1] = np.inf

        if end_disk < kD:
            mult[end_disk + 1 :, j - 1] = 0.0

    return mult


def solve_tensions_pso_tf(
    model,
    cache,
    pso_cfg: dict[str, Any],
    rng_seed: int,
) -> PsoResult:
    try:
        import tensorflow as tf
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("TensorFlow backend requested but tensorflow import failed") from exc

    n_particles = int(pso_cfg["n_particles"])
    iters = int(pso_cfg["iters"])
    inertia = float(pso_cfg["inertia"])
    c1 = float(pso_cfg["c1"])
    c2 = float(pso_cfg["c2"])
    w_resid = float(pso_cfg["w_resid"])
    lambda_max = float(pso_cfg["lambda_max"])
    early = float(pso_cfg["early_stop_mean_rnorm2"])
    objective = str(pso_cfg.get("objective", "weighted_sum")).strip().lower()
    if objective not in {"weighted_sum", "paper_constraint"}:
        raise ValueError(f"Unsupported pso.objective={objective!r}")
    w_tension_mean = float(pso_cfg.get("w_tension_mean", 0.0))
    w_tension_soft_cap = float(pso_cfg.get("w_tension_soft_cap", 0.0))
    tension_soft_cap_ratio = float(pso_cfg.get("tension_soft_cap_ratio", 1.0))
    max_tension_power = float(pso_cfg.get("max_tension_power", 1.0))
    normalized_objectives = bool(pso_cfg.get("normalized_objectives", False))
    tf_device = str(pso_cfg.get("tf_device", "/CPU:0"))
    tf_dtype_name = str(pso_cfg.get("tf_dtype", "float64")).strip().lower()
    tf_xla = bool(pso_cfg.get("tf_xla", False))

    tmin = float(model.t_min)
    tmax = float(model.t_max)
    mu_shaft = float(model.mu_shaft)
    r_shaft_m = float(model.r_shaft_m)
    tau0 = float(model.tau0)
    kD = int(model.kD)

    active = np.asarray(cache.active_ij[1:], dtype=float)  # (kD,12)
    dir_i = np.asarray(cache.dir_i_j[1:], dtype=float) * active[:, :, None]  # (kD,12,3)
    mcoef = np.asarray(cache.mcoef_z[1:], dtype=float) * active  # (kD,12)
    grav_force = np.asarray(cache.grav_force_sum[1:], dtype=float)  # (kD,3)
    grav_mz = np.asarray(cache.grav_mz_sum[1:], dtype=float)  # (kD,)
    bnd_mz = np.asarray(cache.bnd_mz[1:], dtype=float)  # (kD,)
    mult_np = _build_transmission_multipliers(
        theta_rad=cache.theta_rad,
        case_flag_12=cache.case_flag_12,
        end_disk_by_j=model.end_disk_by_j,
        mu_cable=float(model.mu_cable),
        kD=kD,
    )  # (kD+1,12)

    def _run_on_device(device: str) -> PsoResult:
        if tf_dtype_name in {"float32", "fp32", "single"}:
            dtype = tf.float32
        elif tf_dtype_name in {"float64", "fp64", "double"}:
            dtype = tf.float64
        else:
            raise ValueError(f"Unsupported pso.tf_dtype={tf_dtype_name}, expected float32|float64")
        rng = tf.random.Generator.from_seed(int(rng_seed))
        inf_scalar = tf.constant(np.inf, dtype=dtype)

        with tf.device(device):
            mult = tf.constant(mult_np, dtype=dtype)
            dir_tf = tf.constant(dir_i, dtype=dtype)
            mcoef_tf = tf.constant(mcoef, dtype=dtype)
            grav_force_tf = tf.constant(grav_force, dtype=dtype)
            grav_mz_tf = tf.constant(grav_mz, dtype=dtype)
            bnd_mz_tf = tf.constant(bnd_mz, dtype=dtype)

            residual_scale = tf.constant(float(pso_cfg.get("residual_cost_scale", 0.045)), dtype=dtype)
            residual_delta = tf.constant(float(pso_cfg.get("residual_cost_delta", 0.8)), dtype=dtype)
            residual_base = tf.constant(float((float(pso_cfg.get("residual_cost_delta", 0.8)) ** 2) * (np.sqrt(1.0 + (1.0 / float(pso_cfg.get("residual_cost_delta", 0.8))) ** 2) - 1.0)), dtype=dtype)
            paper_feasible_rms = float(pso_cfg.get("paper_feasible_rms_rnorm", 0.06))
            if (not np.isfinite(paper_feasible_rms)) or paper_feasible_rms <= 0.0:
                paper_feasible_rms = 0.06
            paper_feasible_rms_tf = tf.constant(paper_feasible_rms, dtype=dtype)

            @tf.function(reduce_retracing=True, jit_compile=tf_xla)
            def _norm_pseudo_huber(value: tf.Tensor, scale: tf.Tensor, delta: tf.Tensor, base: tf.Tensor) -> tf.Tensor:
                z = tf.abs(value) / tf.maximum(scale, tf.constant(1e-12, dtype=dtype))
                return (tf.square(delta) * (tf.sqrt(1.0 + tf.square(z / delta)) - 1.0)) / tf.maximum(base, tf.constant(1e-12, dtype=dtype))

            @tf.function(reduce_retracing=True, jit_compile=tf_xla)
            def cost_batch(X: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                X = tf.cast(X, dtype)
                F = X[:, None, :] * mult[None, :, :]  # (P,kD+1,12)
                Fi = F[:, 1:, :]  # (P,kD,12)

                cable_mz = tf.reduce_sum(Fi * mcoef_tf[None, :, :], axis=2)  # (P,kD)
                cable_force = tf.einsum("pik,ikc->pic", Fi, dir_tf)  # (P,kD,3)
                Fs = -(grav_force_tf[None, :, :] + cable_force)
                Fs_norm = tf.linalg.norm(Fs, axis=2)  # (P,kD)

                Mz_raw = grav_mz_tf[None, :] + cable_mz + bnd_mz_tf[None, :]
                Mz_fric = -tf.sign(Mz_raw) * mu_shaft * Fs_norm * r_shaft_m
                r_z = Mz_raw + Mz_fric
                r_norm = r_z / tau0
                mean_r2 = tf.reduce_mean(tf.square(r_norm), axis=1)  # (P,)
                rms_rnorm = tf.sqrt(mean_r2)

                max_t = tf.reduce_max(X, axis=1)
                t_norm = X / tmax
                max_ratio = max_t / tmax

                if objective == "paper_constraint":
                    residual_ratio = rms_rnorm / paper_feasible_rms_tf
                    feasible_j = max_ratio + tf.constant(1.0e-3, dtype=dtype) * residual_ratio
                    infeasible_j = tf.constant(1.0e6, dtype=dtype) + tf.square(residual_ratio) + max_ratio
                    J = tf.where(rms_rnorm <= paper_feasible_rms_tf, feasible_j, infeasible_j)
                else:
                    max_term = tf.pow(max_ratio, max_tension_power)
                    mean_term = tf.reduce_mean(tf.square(t_norm), axis=1)
                    cap = float(np.clip(tension_soft_cap_ratio, 0.0, 1.0))
                    excess = tf.nn.relu(t_norm - cap)
                    soft_cap_term = tf.reduce_mean(tf.square(excess), axis=1)

                    resid_term = mean_r2
                    if normalized_objectives:
                        resid_term = _norm_pseudo_huber(rms_rnorm, residual_scale, residual_delta, residual_base)

                    J = (
                        w_resid * resid_term
                        + lambda_max * max_term
                        + w_tension_mean * mean_term
                        + w_tension_soft_cap * soft_cap_term
                    )
                finite = tf.math.is_finite(J) & tf.math.is_finite(mean_r2)
                J = tf.where(finite, J, tf.fill(tf.shape(J), inf_scalar))
                mean_r2 = tf.where(finite, mean_r2, tf.fill(tf.shape(mean_r2), inf_scalar))
                return J, mean_r2

            X = rng.uniform(shape=(n_particles, 12), minval=tmin, maxval=tmax, dtype=dtype)
            V = rng.normal(shape=(n_particles, 12), stddev=(tmax - tmin) * 0.05, dtype=dtype)
            X = tf.clip_by_value(X, tmin, tmax)

            J, mean_r2 = cost_batch(X)
            pbest = tf.identity(X)
            pbest_cost = tf.identity(J)
            g_idx = int(tf.argmin(J).numpy())
            gbest = tf.identity(X[g_idx])
            gbest_cost = float(J[g_idx].numpy())
            gbest_mean_r2 = float(mean_r2[g_idx].numpy())
            evals = n_particles

            if gbest_mean_r2 < early:
                gbest_np = gbest.numpy()
                return PsoResult(
                    T_base_12=gbest_np,
                    best_cost=float(gbest_cost),
                    mean_rnorm2=float(gbest_mean_r2),
                    max_tension=float(np.max(gbest_np)),
                    iters_used=0,
                    evals=evals,
                )

            for it in range(1, iters + 1):
                r1 = rng.uniform(shape=(n_particles, 12), minval=0.0, maxval=1.0, dtype=dtype)
                r2 = rng.uniform(shape=(n_particles, 12), minval=0.0, maxval=1.0, dtype=dtype)
                V = inertia * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
                X = tf.clip_by_value(X + V, tmin, tmax)

                J, mean_r2 = cost_batch(X)
                evals += n_particles

                better = J < pbest_cost
                pbest = tf.where(better[:, None], X, pbest)
                pbest_cost = tf.where(better, J, pbest_cost)

                new_idx = int(tf.argmin(J).numpy())
                new_cost = float(J[new_idx].numpy())
                if new_cost < gbest_cost:
                    gbest_cost = new_cost
                    gbest = tf.identity(X[new_idx])
                    gbest_mean_r2 = float(mean_r2[new_idx].numpy())

                if gbest_mean_r2 < early:
                    gbest_np = gbest.numpy()
                    return PsoResult(
                        T_base_12=gbest_np,
                        best_cost=float(gbest_cost),
                        mean_rnorm2=float(gbest_mean_r2),
                        max_tension=float(np.max(gbest_np)),
                        iters_used=it,
                        evals=evals,
                    )

            gbest_np = gbest.numpy()
            return PsoResult(
                T_base_12=gbest_np,
                best_cost=float(gbest_cost),
                mean_rnorm2=float(gbest_mean_r2),
                max_tension=float(np.max(gbest_np)),
                iters_used=iters,
                evals=evals,
            )

    try:
        return _run_on_device(tf_device)
    except Exception:
        # GPU 不可用时自动回退 CPU，避免任务中断
        return _run_on_device("/CPU:0")
