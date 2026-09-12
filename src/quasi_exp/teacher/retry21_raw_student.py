"""Q33 raw Student experiment: task-space supervision without inverse correction."""
from __future__ import annotations

import copy
import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree

from quasi_exp.model.kinematics_tf import forward_xyz_from_beta_tf
from .retry19_direct_student import DirectStudentConfig, build_direct_student, _keras_direct_feature_layer
from .retry12_symmetry import XYZ_COLUMNS, BETA_COLUMNS


@tf.keras.utils.register_keras_serializable(package='quasi_exp')
class AnchoredBeta(tf.keras.layers.Layer):
    def __init__(self, backbone, zero_xyz, bounds, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.zero_xyz = list(zero_xyz)
        self.bounds = list(bounds)
        self.anchor_enabled = self.add_weight(name='anchor_enabled', shape=(), initializer='zeros', trainable=False)

    def call(self, xyz):
        h = self.backbone(xyz)
        h0 = self.backbone(tf.constant([self.zero_xyz], dtype=xyz.dtype))
        return tf.constant(self.bounds, h.dtype) * tf.tanh(h - self.anchor_enabled * h0)

    def get_config(self):
        return {**super().get_config(), 'backbone': tf.keras.utils.serialize_keras_object(self.backbone),
                'zero_xyz': self.zero_xyz, 'bounds': self.bounds}

    @classmethod
    def from_config(cls, config):
        config['backbone'] = tf.keras.utils.deserialize_keras_object(config['backbone'])
        return cls(**config)


def build_student(normalization_xyz, env, seed):
    tf.keras.utils.set_random_seed(seed)
    base = build_direct_student(train_xyz_m=normalization_xyz, zero_x_m=float(env.fk(np.zeros(6))[0, 0]),
        beta_bounds_rad=env.bounds, config=DirectStudentConfig(signed_power_alpha=1.), include_signed_power=False)
    old = base.get_layer('beta_latent')
    linear = tf.keras.layers.Dense(6, name='beta_logits')
    logits = linear(old.input)
    linear.set_weights(old.get_weights())
    backbone = tf.keras.Model(base.input, logits, name='signed_xyz_backbone')
    if not np.allclose(env.bounds[:, 0], -env.bounds[:, 1]):
        raise ValueError('Q33 requires symmetric beta bounds')
    xyz = tf.keras.Input((3,), name='xyz_m')
    output = AnchoredBeta(backbone, [float(env.fk(np.zeros(6))[0, 0]), 0., 0.], env.bounds[:, 1].tolist(), name='anchored_beta')(xyz)
    return tf.keras.Model(xyz, output, name='retry21_raw_student')


def load_student(path):
    _keras_direct_feature_layer()
    return tf.keras.models.load_model(path, compile=False)


def regions(frame, zero_x):
    xyz = frame[list(XYZ_COLUMNS)].to_numpy(float)
    u = (zero_x - xyz[:, 0]) * 1000
    rho = np.linalg.norm(xyz[:, 1:], axis=1) * 1000
    return {'all': np.ones(len(frame), bool),
            'core': (u >= 60) & (u <= 200) & (rho <= 80),
            'transition': (u >= 60) & (u <= 200) & (rho > 80) & (rho <= 280),
            'tip': (u >= -1e-6) & (u < 60) & (rho <= 160),
            'outer': frame.domain_class.eq('retry18_outer').to_numpy() if 'domain_class' in frame else np.zeros(len(frame), bool)}


class TrainSampler:
    def __init__(self, train, zero_x, seed):
        if len(train) < 1024 or not train.split_role.eq('train').all():
            raise ValueError('sampling requires at least 1024 train-only rows')
        self.count = len(train)
        self.pools = {k: np.flatnonzero(v) for k, v in regions(train, zero_x).items()}
        self.uniform = np.random.default_rng(seed)
        self.stratified = np.random.default_rng(seed + 10000)

    def draw(self, mode):
        uniform = self.uniform.choice(self.count, 1024, replace=False)
        if mode == 'uniform':
            return uniform
        allocations = {'core': 256} if mode == 'core' else {'core': 205, 'transition': 256, 'tip': 154}
        if mode not in ('core', 'regions'):
            raise ValueError('unknown sampling mode')
        parts = [self.stratified.choice(self.pools[k], n, replace=False) for k, n in allocations.items()]
        return np.concatenate(parts + [uniform[:1024 - sum(allocations.values())]])

    def state(self):
        return copy.deepcopy({'uniform': self.uniform.bit_generator.state, 'stratified': self.stratified.bit_generator.state})

    def restore(self, state):
        self.uniform.bit_generator.state = copy.deepcopy(state['uniform'])
        self.stratified.bit_generator.state = copy.deepcopy(state['stratified'])


def loss_terms(prediction, labels, xyz, weight, env):
    coordinate = tf.constant([16., 16., 4., 4., 1., 1.], prediction.dtype)
    weight = tf.cast(weight, prediction.dtype)
    denom = tf.reduce_sum(weight)
    tf.debugging.assert_positive(denom)
    beta_loss = tf.reduce_sum(tf.reduce_sum(coordinate * (prediction - labels)**2, axis=1) * weight) / denom
    actual = forward_xyz_from_beta_tf(prediction, lengths_m=env.lengths_m,
                                     p_end_local_m=env.p_end_local_m, theta_sign=env.theta_sign)
    task_loss = tf.reduce_sum(tf.reduce_sum(((actual - xyz) / .1)**2, axis=1) * weight) / denom
    return beta_loss, task_loss


def make_update(model, optimizer, env, task_loss):
    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32), tf.TensorSpec([None, 6], tf.float32), tf.TensorSpec([None], tf.float32)])
    def update(x, y, w):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            if task_loss:
                beta, task = loss_terms(pred, y, x, w, env)
                loss = .01 * beta + task
            else:
                beta = tf.reduce_sum(tf.reduce_sum(tf.constant([16.,16.,4.,4.,1.,1.])*(pred-y)**2, axis=1)*w)/tf.reduce_sum(w)
                task = tf.constant(0.)
                loss = beta
            tf.debugging.assert_all_finite(loss, 'nonfinite training loss')
        gradients = tape.gradient(loss, model.trainable_variables)
        for g in gradients:
            tf.debugging.assert_all_finite(g, 'nonfinite gradient')
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, beta, task
    return update


def predict(model, xyz, chunk=4096):
    return np.concatenate([np.asarray(model(np.asarray(xyz[i:i+chunk], np.float32), training=False), float)
                           for i in range(0, len(xyz), chunk)])


def error_metrics(error):
    error = np.asarray(error, float)
    finite = error[np.isfinite(error)]
    return {'count': len(error), 'finite_count': len(finite), 'nonfinite_count': int((~np.isfinite(error)).sum()),
            **{f'p{p}_mm': float(np.percentile(finite, p)) if len(finite) else None for p in (50, 95, 99)},
            'max_mm': float(finite.max()) if len(finite) else None,
            'within_3mm': float(np.mean(error <= 3)) if len(error) else None,
            'within_10mm': float(np.mean(error <= 10)) if len(error) else None}


def neighbor_audit(train, targets, env):
    xyz = targets[list(XYZ_COLUMNS)].to_numpy(float)
    tx = train[list(XYZ_COLUMNS)].to_numpy(float)
    tb = train[list(BETA_COLUMNS)].to_numpy(float)
    distance, ids = cKDTree(tx).query(xyz, k=32)
    rows = []
    nearest_norm = np.linalg.norm(tb[ids[:, 0]], axis=1)
    for k in (1, 2, 4, 8, 16, 32):
        b = tb[ids[:, :k]].mean(axis=1)
        actual, jac = env.fk_and_jacobian(b)
        singular = np.linalg.svd(jac, compute_uv=False)
        condition = singular[:, 0] / np.maximum(singular[:, -1], 1e-15)
        for i in range(len(xyz)):
            rows.append({'target_id': str(targets.iloc[i].target_id), 'k': k,
                         'fk_error_mm': float(np.linalg.norm(actual[i] - xyz[i]) * 1000),
                         'neighbor_distance_mm': float(distance[i, k-1]*1000),
                         'beta_norm_ratio': float(np.linalg.norm(b[i])/max(nearest_norm[i], 1e-15)),
                         'jacobian_condition': float(condition[i])})
    return rows
