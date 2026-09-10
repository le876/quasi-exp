"""Anchored relocation diagnostic: labels, paired rows and fixed-step students."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from .retry12_symmetry import XYZ_COLUMNS, BETA_COLUMNS
from .retry19_direct_student import DirectStudentConfig, build_direct_student, signed_macroblock_ids

WEIGHTS = np.array([4., 4., 2., 2., 1., 1.])


def bounded_dls(env, xyz, initial, *, steps=20, damping=.002, maximum_step_deg=3.):
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    beta = np.asarray(initial, float).reshape(-1, 6).copy()
    bounds = np.asarray(env.bounds)
    beta = np.clip(beta, bounds[:, 0], bounds[:, 1])
    for _ in range(steps):
        actual, jac = env.fk_and_jacobian(beta)
        error = xyz - np.asarray(actual).reshape(-1, 3)
        if np.max(np.linalg.norm(error, axis=1), initial=0.) < 1e-8:
            break
        jac = np.asarray(jac).reshape(-1, 3, 6)
        jw = jac / WEIGHTS
        task = jw @ np.swapaxes(jac, 1, 2) + damping**2 * np.eye(3)
        delta = np.einsum('nij,nj->ni', np.swapaxes(jw, 1, 2), np.linalg.solve(task, error[..., None])[..., 0])
        rms = np.sqrt(np.mean(delta**2, axis=1))
        delta *= np.minimum(1., np.deg2rad(maximum_step_deg) / np.maximum(rms, 1e-15))[:, None]
        beta = np.clip(beta + delta, bounds[:, 0], bounds[:, 1])
    return beta


def solve_anchored(env, xyz, anchors, seed_bank_xyz, seed_bank_beta, *, fallback_seeds=16, deadline=float('inf')):
    """Two local continuations; bounded LS fallback, selected toward old anchor.

    This is an anchored diagnostic Teacher, not global graph canonicalization.
    Proposal joint parameters are never consumed.
    """
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    anchors = np.asarray(anchors, float).reshape(-1, 6)
    bank = np.asarray(seed_bank_beta, float)
    _, nearest = cKDTree(seed_bank_xyz).query(xyz, k=min(fallback_seeds + 1, len(bank)))
    nearest = np.asarray(nearest).reshape(len(xyz), -1)
    second = []
    for anchor, ids in zip(anchors, nearest):
        distinct = [i for i in ids if np.linalg.norm(bank[i] - anchor) > 1e-8]
        if not distinct:
            raise ValueError('no distinct canonical seed')
        second.append(bank[distinct[0]])
    solutions = [bounded_dls(env, xyz, seeds) for seeds in (anchors, np.asarray(second))]
    residuals = [np.linalg.norm(env.fk(b) - xyz, axis=1) * 1000 for b in solutions]
    gap = np.rad2deg(np.sqrt(np.mean(((solutions[0] - solutions[1])*WEIGHTS)**2, axis=1)))
    raw_gap = np.rad2deg(np.max(np.abs(solutions[0] - solutions[1]), axis=1))
    fast = (residuals[0] <= 3) & (residuals[1] <= 3) & (gap <= 2) & (raw_gap <= 5)
    bounds = np.asarray(env.bounds)
    output = np.full_like(anchors, np.nan)
    rows = []
    for n, target in enumerate(xyz):
        choices = [(b[n], r[n], 'two_seed_dls') for b, r in zip(solutions, residuals) if r[n] <= 3]
        fallback_count = 0
        timed_out = False
        if not fast[n]:
            # Solve all bounded seeds to avoid making branch selection depend on the first success.
            for seed in bank[nearest[n, :fallback_seeds]]:
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                fit = least_squares(lambda b: np.asarray(env.fk(b)).reshape(3)-target,
                    np.clip(seed, bounds[:, 0]+1e-12, bounds[:, 1]-1e-12),
                    jac=lambda b: env.jacobian(b), bounds=(bounds[:, 0], bounds[:, 1]),
                    max_nfev=200, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                fallback_count += 1
                residual = float(np.linalg.norm(env.fk(fit.x).reshape(3)-target)*1000)
                if residual <= 3:
                    choices.append((fit.x, residual, 'bounded_least_squares'))
        if choices:
            chosen = min(choices, key=lambda c: (float(np.sum(((c[0]-anchors[n])*WEIGHTS)**2)), c[1]))
            output[n] = chosen[0]
            residual, method = float(chosen[1]), chosen[2]
        else:
            residual, method = np.nan, 'unresolved'
        rows.append(dict(accepted=bool(choices), fast_gold=bool(fast[n]), residual_mm=residual,
            two_seed_weighted_gap_deg=float(gap[n]), two_seed_raw_gap_deg=float(raw_gap[n]),
            fallback_seed_count=fallback_count, solver=method, budget_exhausted=timed_out))
    return output, pd.DataFrame(rows)


def paired_datasets(original, prospective, replacements, solved_beta, accepted):
    if original.target_id.duplicated().any() or prospective.target_id.duplicated().any() or replacements.target_id.duplicated().any():
        raise ValueError('duplicate row identity')
    original = original.reset_index(drop=True)
    ids = pd.Index(original.target_id)
    at = ids.get_indexer(replacements.target_id)
    if np.any(at < 0):
        raise ValueError('replacement missing from original')
    p = prospective.set_index('target_id').loc[replacements.target_id]
    if not np.array_equal(p[list(XYZ_COLUMNS)].to_numpy(), replacements[list(XYZ_COLUMNS)].to_numpy()):
        raise ValueError('candidate layout changed')
    old = original.iloc[at]
    if not np.array_equal(old.split_role.to_numpy(), replacements.split_role.to_numpy()):
        raise ValueError('split changed')
    if not np.array_equal(signed_macroblock_ids(old).to_numpy(), signed_macroblock_ids(replacements).to_numpy()):
        raise ValueError('replacement crossed macroblock')
    accepted = np.asarray(accepted, bool)
    beta = np.asarray(solved_beta, float)
    if beta.shape != (len(at), 6) or not np.isfinite(beta[accepted]).all():
        raise ValueError('invalid accepted labels')
    columns = ['target_id', *XYZ_COLUMNS, *BETA_COLUMNS, 'split_role', 'sample_weight', 'domain_class']
    left = original[columns].copy()
    right = left.copy()
    right.loc[at, list(XYZ_COLUMNS)] = replacements[list(XYZ_COLUMNS)].to_numpy()
    right.loc[at, list(BETA_COLUMNS)] = beta
    keep = np.ones(len(left), bool)
    keep[at[~accepted]] = False
    # The paired baseline uses the same row mask; failed labels never fall back to old coordinates.
    return left.loc[keep].reset_index(drop=True), right.loc[keep].reset_index(drop=True), keep


def prediction_metrics(env, xyz, beta):
    xyz = np.asarray(xyz, float)
    beta = np.asarray(beta, float)
    finite = np.isfinite(beta).all(axis=1)
    actual = np.full((len(beta), 3), np.nan)
    if finite.any():
        actual[finite] = env.fk(beta[finite])
    residual = np.linalg.norm(actual - xyz, axis=1)*1000
    valid_error = residual[finite]
    bounds = np.asarray(env.bounds)
    violation = finite & ((beta < bounds[:, 0]-1e-10) | (beta > bounds[:, 1]+1e-10)).any(axis=1)
    metrics = dict(count=len(beta), finite_count=int(finite.sum()), failed_count=int((~finite).sum()),
        fk_p50_mm=float(np.percentile(valid_error, 50)) if len(valid_error) else None,
        fk_p95_mm=float(np.percentile(valid_error, 95)) if len(valid_error) else None,
        fk_max_mm=float(np.max(valid_error)) if len(valid_error) else None,
        within_3mm_rate=float(np.mean(finite & (residual <= 3))),
        within_10mm_rate=float(np.mean(finite & (residual <= 10))), bounds_violation_count=int(violation.sum()),
        quantile_denominator='finite predictions; success rates use all targets')
    return metrics, actual, residual


def train_pair(left, right, env, panel_validation, output, write_json, *, steps=500, seed=20260925):
    import tensorflow as tf
    from pathlib import Path
    output = Path(output)
    a = left[left.split_role.eq('train')].reset_index(drop=True)
    b = right[right.split_role.eq('train')].reset_index(drop=True)
    valid = left[left.split_role.eq('validation')]
    if a.empty or valid.empty or not a.target_id.equals(b.target_id):
        raise ValueError('invalid paired train/validation rows')
    cfg = DirectStudentConfig(maximum_steps=steps, seed=seed, signed_power_alpha=1., validation_interval=25)
    tf.keras.utils.set_random_seed(seed)
    first = build_direct_student(train_xyz_m=a[list(XYZ_COLUMNS)].to_numpy(), zero_x_m=float(env.fk(np.zeros(6))[0, 0]),
        beta_bounds_rad=env.bounds, config=cfg, include_signed_power=False)
    initial = first.get_weights()
    weights = a.sample_weight.fillna(1.).to_numpy(np.float32)
    histories, models = [], {}
    for name, train in [('L0', a), ('L1', b)]:
        model = build_direct_student(train_xyz_m=a[list(XYZ_COLUMNS)].to_numpy(), zero_x_m=float(env.fk(np.zeros(6))[0, 0]),
            beta_bounds_rad=env.bounds, config=cfg, include_signed_power=False)
        model.set_weights(initial)
        optimizer = tf.keras.optimizers.Adam(cfg.learning_rate)
        xyz, labels = train[list(XYZ_COLUMNS)].to_numpy(np.float32), train[list(BETA_COLUMNS)].to_numpy(np.float32)
        coordinate = tf.constant(WEIGHTS.astype(np.float32)**2)
        @tf.function
        def update(x, y, w):
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(tf.reduce_sum(coordinate*(model(x, training=True)-y)**2, axis=1)*w)/tf.reduce_sum(w)
            gradient = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            return loss
        rng = np.random.default_rng(seed)
        history = []
        for step in range(1, steps+1):
            batch = rng.choice(len(train), min(1024, len(train)), replace=False)
            loss = update(xyz[batch], labels[batch], weights[batch])
            if step % 25 == 0 or step == steps:
                row = {'model': name, 'step': step, 'train_beta_loss': float(loss)}
                for role, vx in [('on_grid', valid[list(XYZ_COLUMNS)].to_numpy()), ('off_grid', panel_validation)]:
                    pred = np.asarray(model(vx.astype(np.float32), training=False), float)
                    metric, _, _ = prediction_metrics(env, vx, pred)
                    row[role+'_fk_p95_mm'] = metric['fk_p95_mm']
                history.append(row)
                write_json(output/'training_progress.json', row)
        model.save(output/(name+'.keras'))
        histories.extend(history)
        models[name] = model
    pd.DataFrame(histories).to_parquet(output/'training_history.parquet', index=False)
    write_json(output/'training_contract.json', dict(seed=seed, optimizer_steps=steps, selection='fixed_final_step',
        architecture=[128,128,64], normalization='same L0 train features', initialization='identical weights',
        minibatches='same ordered row indices', failed_relocation_rows_removed_from_both=True,
        radial_scale_mm=cfg.radial_scale_mm, axial_scale_mm=cfg.axial_scale_mm, learning_rate=cfg.learning_rate))
    return models
