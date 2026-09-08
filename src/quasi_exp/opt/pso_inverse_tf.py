from __future__ import annotations

from typing import Any

import numpy as np

from quasi_exp.model.sampling import beta_to_theta


def _theta_index_map_30() -> np.ndarray:
    idx = np.zeros(30, dtype=int)
    for i in range(1, 31):
        section = (i - 1) // 10
        is_odd = (i % 2) == 1
        if section == 0:
            idx[i - 1] = 0 if is_odd else 1
        elif section == 1:
            idx[i - 1] = 2 if is_odd else 3
        else:
            idx[i - 1] = 4 if is_odd else 5
    return idx


def _alpha_array_30() -> np.ndarray:
    a = np.zeros(30, dtype=float)
    for i in range(1, 31):
        if i == 1:
            a[i - 1] = 0.0
        else:
            a[i - 1] = (np.pi / 2.0) if (i % 2 == 1) else (-np.pi / 2.0)
    return a


def solve_inverse_joint_pso_tf(
    model,
    inputs,
    xyz_target_m: np.ndarray,
    beta_ranges_rad: dict[str, Any],
    inverse_pso_cfg: dict[str, Any],
    tension_pso_cfg: dict[str, Any],
    rng_seed: int,
    warm_start_beta: np.ndarray | None = None,
    continuity_ref_beta: np.ndarray | None = None,
):
    try:
        import tensorflow as tf
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("TensorFlow backend requested but tensorflow import failed") from exc

    from .pso_inverse import _canonicalize_beta_from_candidates, _finalize_result

    n_particles = int(inverse_pso_cfg["n_particles"])
    iters = int(inverse_pso_cfg["iters"])
    inertia = float(inverse_pso_cfg["inertia"])
    c1 = float(inverse_pso_cfg["c1"])
    c2 = float(inverse_pso_cfg["c2"])
    w_xyz = float(inverse_pso_cfg["w_xyz"])
    w_beta_l2 = float(inverse_pso_cfg.get("w_beta_l2", 0.0))
    early_xyz = float(inverse_pso_cfg.get("early_stop_xyz_err_m", 0.0))
    w_cont = float(inverse_pso_cfg.get("w_continuity", 0.0))
    cont_ratio_cap = float(np.clip(inverse_pso_cfg.get("continuity_max_ratio", 0.3), 0.0, 1.0))
    cont_relax_if_err_ratio = float(inverse_pso_cfg.get("continuity_relax_if_err_ratio", 1.2))
    cont_relax_scale = float(np.clip(inverse_pso_cfg.get("continuity_relax_scale", 0.3), 0.0, 1.0))
    warm_start_enabled = bool(inverse_pso_cfg.get("warm_start", True))
    warm_start_particles = max(1, int(inverse_pso_cfg.get("warm_start_particles", 6)))
    warm_start_sigma = max(0.0, float(inverse_pso_cfg.get("warm_start_sigma", 0.08)))
    tf_device = str(inverse_pso_cfg.get("tf_device", "/CPU:0"))
    tf_dtype_name = str(inverse_pso_cfg.get("tf_dtype", "float64")).strip().lower()
    tf_xla = bool(inverse_pso_cfg.get("tf_xla", False))
    normalized_objectives = bool(inverse_pso_cfg.get("normalized_objectives", False))

    beta_bounds = np.zeros((6, 2), dtype=float)
    for i in range(6):
        lo, hi = beta_ranges_rad[f"beta{i+1}"]
        beta_bounds[i, 0] = float(lo)
        beta_bounds[i, 1] = float(hi)
    beta_lo = beta_bounds[:, 0]
    beta_hi = beta_bounds[:, 1]
    beta_scale = np.maximum(np.maximum(np.abs(beta_lo), np.abs(beta_hi)), 1e-9)
    continuity_ref = None
    if continuity_ref_beta is not None:
        continuity_ref = np.asarray(continuity_ref_beta, dtype=float).reshape(6)
    target = np.asarray(xyz_target_m, dtype=float).reshape(3)

    lengths = np.asarray(inputs.lengths_m, dtype=float)
    p_end = np.asarray(inputs.p_end_local_m, dtype=float)
    theta_sign = float(model.theta_sign)
    theta_map = _theta_index_map_30()
    alpha = _alpha_array_30()
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    a_len = lengths[:-1].copy()

    def _run_on_device(device: str):
        if tf_dtype_name in {"float32", "fp32", "single"}:
            dtype = tf.float32
        elif tf_dtype_name in {"float64", "fp64", "double"}:
            dtype = tf.float64
        else:
            raise ValueError(
                f"Unsupported inverse_pso.tf_dtype={tf_dtype_name}, expected float32|float64"
            )
        rng = tf.random.Generator.from_seed(int(rng_seed))
        inf_scalar = tf.constant(np.inf, dtype=dtype)

        with tf.device(device):
            beta_lo_tf = tf.constant(beta_lo, dtype=dtype)
            beta_hi_tf = tf.constant(beta_hi, dtype=dtype)
            beta_scale_tf = tf.constant(beta_scale, dtype=dtype)
            continuity_ref_tf = None
            if continuity_ref is not None:
                continuity_ref_tf = tf.constant(continuity_ref, dtype=dtype)
            target_tf = tf.constant(target, dtype=dtype)
            p_end_tf = tf.constant(p_end, dtype=dtype)
            theta_map_tf = tf.constant(theta_map, dtype=tf.int32)
            ca_tf = tf.constant(ca, dtype=dtype)
            sa_tf = tf.constant(sa, dtype=dtype)
            a_tf = tf.constant(a_len, dtype=dtype)

            @tf.function(reduce_retracing=True, jit_compile=tf_xla)
            def forward_xyz_from_beta_batch(beta6: tf.Tensor) -> tf.Tensor:
                beta6 = tf.cast(beta6, dtype)
                theta = tf.gather(beta6, theta_map_tf, axis=1) * theta_sign  # (P,30)
                pnum = tf.shape(beta6)[0]
                T = tf.tile(tf.eye(4, dtype=dtype)[None, :, :], [pnum, 1, 1])
                z = tf.zeros((pnum,), dtype=dtype)
                o = tf.ones((pnum,), dtype=dtype)
                for i in range(30):
                    ct = tf.cos(theta[:, i])
                    st = tf.sin(theta[:, i])
                    ai = a_tf[i]
                    cai = ca_tf[i]
                    sai = sa_tf[i]
                    row0 = tf.stack([ct, -st, z, tf.fill(tf.shape(ct), ai)], axis=1)
                    row1 = tf.stack([st * cai, ct * cai, tf.fill(tf.shape(ct), -sai), z], axis=1)
                    row2 = tf.stack([st * sai, ct * sai, tf.fill(tf.shape(ct), cai), z], axis=1)
                    row3 = tf.stack([z, z, z, o], axis=1)
                    Ti = tf.stack([row0, row1, row2, row3], axis=1)  # (P,4,4)
                    T = tf.matmul(T, Ti)
                p = tf.linalg.matvec(T, p_end_tf)  # (P,4)
                return p[:, :3]

            xyz_scale = tf.constant(float(inverse_pso_cfg.get("xyz_cost_scale_m", 0.015)), dtype=dtype)
            xyz_delta = tf.constant(float(inverse_pso_cfg.get("xyz_cost_delta", 0.6)), dtype=dtype)
            cont_scale_deg = tf.constant(float(inverse_pso_cfg.get("cont_cost_scale_deg", 6.0)), dtype=dtype)
            cont_delta = tf.constant(float(inverse_pso_cfg.get("cont_cost_delta", 0.4)), dtype=dtype)
            deg_per_rad = tf.constant(180.0 / np.pi, dtype=dtype)
            xyz_base = tf.constant(float((float(inverse_pso_cfg.get("xyz_cost_delta", 0.6)) ** 2) * (np.sqrt(1.0 + (1.0 / float(inverse_pso_cfg.get("xyz_cost_delta", 0.6))) ** 2) - 1.0)), dtype=dtype)
            cont_base = tf.constant(float((float(inverse_pso_cfg.get("cont_cost_delta", 0.4)) ** 2) * (np.sqrt(1.0 + (1.0 / float(inverse_pso_cfg.get("cont_cost_delta", 0.4))) ** 2) - 1.0)), dtype=dtype)

            @tf.function(reduce_retracing=True, jit_compile=tf_xla)
            def _norm_pseudo_huber(value: tf.Tensor, scale: tf.Tensor, delta: tf.Tensor, base: tf.Tensor) -> tf.Tensor:
                z = tf.abs(value) / tf.maximum(scale, tf.constant(1e-12, dtype=dtype))
                return (tf.square(delta) * (tf.sqrt(1.0 + tf.square(z / delta)) - 1.0)) / tf.maximum(base, tf.constant(1e-12, dtype=dtype))

            @tf.function(reduce_retracing=True, jit_compile=tf_xla)
            def cost_batch(X: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                X = tf.cast(X, dtype)
                p_xyz = forward_xyz_from_beta_batch(X)
                xyz_err = tf.linalg.norm(p_xyz - target_tf[None, :], axis=1)
                beta_term = tf.reduce_mean(tf.square(X / beta_scale_tf[None, :]), axis=1)
                if normalized_objectives:
                    xyz_term = _norm_pseudo_huber(xyz_err, xyz_scale, xyz_delta, xyz_base)
                    base_cost = w_xyz * xyz_term + w_beta_l2 * beta_term
                else:
                    base_cost = w_xyz * tf.square(xyz_err) + w_beta_l2 * beta_term
                if (continuity_ref_tf is not None) and (w_cont > 0.0):
                    if normalized_objectives:
                        delta_rms_deg = tf.sqrt(tf.reduce_mean(tf.square(X - continuity_ref_tf[None, :]), axis=1)) * deg_per_rad
                        delta_term = _norm_pseudo_huber(delta_rms_deg, cont_scale_deg, cont_delta, cont_base)
                    else:
                        delta_term = tf.reduce_mean(tf.square((X - continuity_ref_tf[None, :]) / beta_scale_tf[None, :]), axis=1)
                    eff_w = tf.fill(tf.shape(delta_term), tf.constant(w_cont, dtype=dtype))
                    if early_xyz > 0.0:
                        relax_cond = xyz_err > tf.constant(cont_relax_if_err_ratio * early_xyz, dtype=dtype)
                        eff_w = tf.where(relax_cond, eff_w * tf.constant(cont_relax_scale, dtype=dtype), eff_w)
                    cont_cost = eff_w * delta_term
                    if cont_ratio_cap > 0.0:
                        cap = tf.constant(cont_ratio_cap, dtype=dtype) * tf.maximum(base_cost, tf.constant(1e-12, dtype=dtype))
                        cont_cost = tf.minimum(cont_cost, cap)
                    J = base_cost + cont_cost
                else:
                    J = base_cost
                finite = tf.math.is_finite(J) & tf.math.is_finite(xyz_err)
                J = tf.where(finite, J, tf.fill(tf.shape(J), inf_scalar))
                xyz_err = tf.where(finite, xyz_err, tf.fill(tf.shape(xyz_err), inf_scalar))
                return J, xyz_err

            X = rng.uniform(shape=(n_particles, 6), minval=beta_lo_tf, maxval=beta_hi_tf, dtype=dtype)
            X = tf.tensor_scatter_nd_update(X, indices=[[0]], updates=[tf.zeros((6,), dtype=dtype)])
            vscale = np.maximum(beta_hi - beta_lo, 1e-6) * 0.08
            V = rng.normal(shape=(n_particles, 6), stddev=tf.constant(vscale, dtype=dtype), dtype=dtype)

            if warm_start_enabled and (warm_start_beta is not None):
                warm = np.asarray(warm_start_beta, dtype=float).reshape(6)
                if np.isfinite(warm).all():
                    warm = np.clip(warm, beta_lo, beta_hi)
                    warm_tf = tf.constant(warm, dtype=dtype)
                    X = tf.tensor_scatter_nd_update(X, indices=[[0]], updates=[warm_tf])
                    warm_n = min(n_particles, warm_start_particles)
                    if warm_n > 1:
                        stddev = tf.constant(vscale * warm_start_sigma, dtype=dtype)
                        noise = rng.normal(shape=(warm_n - 1, 6), stddev=stddev, dtype=dtype)
                        cloud = tf.clip_by_value(warm_tf[None, :] + noise, beta_lo_tf[None, :], beta_hi_tf[None, :])
                        idx = tf.reshape(tf.range(1, warm_n, dtype=tf.int32), (-1, 1))
                        X = tf.tensor_scatter_nd_update(X, indices=idx, updates=cloud)

            J, xyz_err = cost_batch(X)
            pbest = tf.identity(X)
            pbest_cost = tf.identity(J)
            g_idx = int(tf.argmin(J).numpy())
            gbest = tf.identity(X[g_idx])
            gbest_cost = float(J[g_idx].numpy())
            gbest_xyz = float(xyz_err[g_idx].numpy())
            evals = n_particles

            stop_iter = 0
            if gbest_xyz <= early_xyz:
                return gbest.numpy(), pbest.numpy(), gbest_cost, stop_iter, evals

            for it in range(1, iters + 1):
                r1 = rng.uniform(shape=(n_particles, 6), minval=0.0, maxval=1.0, dtype=dtype)
                r2 = rng.uniform(shape=(n_particles, 6), minval=0.0, maxval=1.0, dtype=dtype)
                V = inertia * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
                X = tf.clip_by_value(X + V, beta_lo_tf[None, :], beta_hi_tf[None, :])

                J, xyz_err = cost_batch(X)
                evals += n_particles

                better = J < pbest_cost
                pbest = tf.where(better[:, None], X, pbest)
                pbest_cost = tf.where(better, J, pbest_cost)

                new_idx = int(tf.argmin(J).numpy())
                new_cost = float(J[new_idx].numpy())
                if new_cost < gbest_cost:
                    gbest_cost = new_cost
                    gbest = tf.identity(X[new_idx])
                    gbest_xyz = float(xyz_err[new_idx].numpy())

                if gbest_xyz <= early_xyz:
                    stop_iter = it
                    return gbest.numpy(), pbest.numpy(), gbest_cost, stop_iter, evals

            stop_iter = iters
            return gbest.numpy(), pbest.numpy(), gbest_cost, stop_iter, evals

    try:
        gbest_np, pbest_np, gbest_cost, stop_iter, evals = _run_on_device(tf_device)
    except Exception:
        gbest_np, pbest_np, gbest_cost, stop_iter, evals = _run_on_device("/CPU:0")

    mode = str(inverse_pso_cfg.get("canonical_mode", "none")).strip().lower()
    if mode in {"paper_minmax_tension"}:
        cand = np.vstack([gbest_np.reshape(1, 6), pbest_np])
        gbest_np = _canonicalize_beta_from_candidates(
            model=model,
            inputs=inputs,
            target_xyz_m=target,
            candidate_betas=cand,
            inverse_pso_cfg=inverse_pso_cfg,
            tension_pso_cfg=tension_pso_cfg,
            rng_seed=int(rng_seed),
            continuity_ref_beta=continuity_ref,
        )

    return _finalize_result(
        model=model,
        inputs=inputs,
        target_xyz_m=target,
        best_beta=gbest_np,
        best_cost=float(gbest_cost),
        iters_used=int(stop_iter),
        evals=int(evals),
        tension_pso_cfg=tension_pso_cfg,
        rng_seed=int(rng_seed),
    )
