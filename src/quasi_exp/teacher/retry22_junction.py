"""Paired interventions for Q33 junction sampling and beta-label tether."""
from __future__ import annotations

import copy
import numpy as np
import tensorflow as tf

from .retry21_raw_student import TrainSampler, loss_terms


class JunctionSampler(TrainSampler):
    """Keep the original random streams; replace 128 uniform rows in one arm."""

    def __init__(self, train, zero_x, seed, halfwidth_mm=60.0, count=128):
        super().__init__(train, zero_x, seed)
        self.junction = np.flatnonzero(np.abs(train.x_m.to_numpy() * 1000 - (zero_x * 1000 - 200)) <= halfwidth_mm)
        self.count_junction = count
        if len(self.junction) < count or not 0 < count <= 409:
            raise ValueError('insufficient train-only junction pool or invalid allocation')
        self.junction_rng = np.random.default_rng(seed + 20000)

    def draw(self, mode):
        if mode != 'junction':
            return super().draw(mode)
        base = super().draw('regions')
        extra = self.junction_rng.choice(self.junction, self.count_junction, replace=False)
        return np.concatenate([base[:-self.count_junction], extra])

    def state(self):
        return {**super().state(), 'junction': copy.deepcopy(self.junction_rng.bit_generator.state)}

    def restore(self, state):
        super().restore(state)
        self.junction_rng.bit_generator.state = copy.deepcopy(state['junction'])


def make_intervention_update(model, optimizer, env, beta_coefficient):
    if beta_coefficient not in (0.0, 0.01):
        raise ValueError('unregistered beta coefficient')

    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32), tf.TensorSpec([None, 6], tf.float32), tf.TensorSpec([None], tf.float32)])
    def update(x, labels, weights):
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            beta, task = loss_terms(prediction, labels, x, weights, env)
            loss = beta_coefficient * beta + task
            tf.debugging.assert_all_finite(loss, 'nonfinite intervention loss')
        gradients = tape.gradient(loss, model.trainable_variables)
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, 'nonfinite intervention gradient')
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, beta, task

    return update
