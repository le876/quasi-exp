#!/usr/bin/env python3
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TFGpuStatus:
    gpu_count: int
    gpu_names: list[str]


def tensorflow_gpu_status() -> TFGpuStatus:
    import tensorflow as tf  # type: ignore

    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [getattr(g, "name", str(g)) for g in gpus]
    return TFGpuStatus(gpu_count=len(gpus), gpu_names=gpu_names)


def configure_tf_runtime(tf_device: str = "auto", require_gpu: bool = False, seed: int = 20260207) -> TFGpuStatus:
    import tensorflow as tf  # type: ignore

    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))

    gpus = tf.config.list_physical_devices("GPU")
    for dev in gpus:
        try:
            tf.config.experimental.set_memory_growth(dev, True)
        except Exception:
            pass

    if tf_device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        gpus = []
    elif tf_device == "gpu" and len(gpus) == 0:
        raise RuntimeError("tf_device=gpu but no visible GPU device")

    if require_gpu and len(gpus) < 1:
        raise RuntimeError("require_gpu is enabled but TensorFlow cannot see any GPU")

    gpu_names = [getattr(g, "name", str(g)) for g in gpus]
    return TFGpuStatus(gpu_count=len(gpus), gpu_names=gpu_names)


class TFDualHeadRegressor:
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (256, 128, 64, 32),
        alpha: float = 1e-6,
        learning_rate_init: float = 1e-3,
        batch_size: int = 1024,
        max_iter: int = 300,
        n_iter_no_change: int = 20,
        tol: float = 1e-6,
        theta_dim: int = 30,
        tension_dim: int = 12,
        loss_lambda_theta: float = 1.0,
        loss_lambda_tension: float = 1.0,
        loss_lambda_phys: float = 0.0,
        tension_lower_norm: np.ndarray | None = None,
        tension_upper_norm: np.ndarray | None = None,
        random_state: int = 20260207,
        validation_split: float = 0.1,
        verbose: int = 1,
    ):
        self.hidden_layer_sizes = tuple(int(v) for v in hidden_layer_sizes)
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.n_iter_no_change = int(n_iter_no_change)
        self.tol = float(tol)
        self.theta_dim = int(theta_dim)
        self.tension_dim = int(tension_dim)
        self.loss_lambda_theta = float(loss_lambda_theta)
        self.loss_lambda_tension = float(loss_lambda_tension)
        self.loss_lambda_phys = float(loss_lambda_phys)
        self.random_state = int(random_state)
        self.validation_split = float(validation_split)
        self.verbose = int(verbose)
        if tension_lower_norm is None:
            tension_lower_norm = np.full((self.tension_dim,), -1e9, dtype=float)
        if tension_upper_norm is None:
            tension_upper_norm = np.full((self.tension_dim,), 1e9, dtype=float)
        self.tension_lower_norm = np.asarray(tension_lower_norm, dtype=float).reshape(self.tension_dim)
        self.tension_upper_norm = np.asarray(tension_upper_norm, dtype=float).reshape(self.tension_dim)

        self.model_ = None
        self.loss_curve_: list[float] = []
        self.validation_scores_: list[float] = []
        self.n_iter_: int = 0

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "batch_size": self.batch_size,
            "max_iter": self.max_iter,
            "n_iter_no_change": self.n_iter_no_change,
            "tol": self.tol,
            "theta_dim": self.theta_dim,
            "tension_dim": self.tension_dim,
            "loss_lambda_theta": self.loss_lambda_theta,
            "loss_lambda_tension": self.loss_lambda_tension,
            "loss_lambda_phys": self.loss_lambda_phys,
            "tension_lower_norm": self.tension_lower_norm.copy(),
            "tension_upper_norm": self.tension_upper_norm.copy(),
            "random_state": self.random_state,
            "validation_split": self.validation_split,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> "TFDualHeadRegressor":
        for key, value in params.items():
            if not hasattr(self, key):
                continue
            setattr(self, key, value)
        if hasattr(self, "hidden_layer_sizes"):
            self.hidden_layer_sizes = tuple(int(v) for v in self.hidden_layer_sizes)
        self.batch_size = int(self.batch_size)
        self.max_iter = int(self.max_iter)
        self.n_iter_no_change = int(self.n_iter_no_change)
        self.theta_dim = int(self.theta_dim)
        self.tension_dim = int(self.tension_dim)
        self.alpha = float(self.alpha)
        self.learning_rate_init = float(self.learning_rate_init)
        self.tol = float(self.tol)
        self.loss_lambda_theta = float(self.loss_lambda_theta)
        self.loss_lambda_tension = float(self.loss_lambda_tension)
        self.loss_lambda_phys = float(self.loss_lambda_phys)
        self.validation_split = float(self.validation_split)
        self.verbose = int(self.verbose)
        self.random_state = int(self.random_state)
        self.tension_lower_norm = np.asarray(self.tension_lower_norm, dtype=float).reshape(self.tension_dim)
        self.tension_upper_norm = np.asarray(self.tension_upper_norm, dtype=float).reshape(self.tension_dim)
        return self

    def _build_model(self, input_dim: int):
        import tensorflow as tf  # type: ignore

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        inputs = tf.keras.Input(shape=(int(input_dim),), name="input")
        x = inputs
        for idx, units in enumerate(self.hidden_layer_sizes):
            x = tf.keras.layers.Dense(
                int(units),
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(self.alpha),
                name=f"dense_{idx+1}",
            )(x)

        theta_out = tf.keras.layers.Dense(self.theta_dim, name="theta")(x)
        tension_out = tf.keras.layers.Dense(self.tension_dim, name="tension")(x)

        lower = tf.constant(self.tension_lower_norm.reshape(1, self.tension_dim), dtype=tf.float32)
        upper = tf.constant(self.tension_upper_norm.reshape(1, self.tension_dim), dtype=tf.float32)
        lambda_phys = tf.constant(self.loss_lambda_phys, dtype=tf.float32)

        def tension_loss(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
            if self.loss_lambda_phys <= 0.0:
                return mse
            below = tf.nn.relu(lower - y_pred)
            above = tf.nn.relu(y_pred - upper)
            penalty = tf.reduce_mean(tf.square(below) + tf.square(above), axis=-1)
            return mse + lambda_phys * penalty

        model = tf.keras.Model(inputs=inputs, outputs={"theta": theta_out, "tension": tension_out}, name="tf_dual_head_regressor")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate_init),
            loss={"theta": "mse", "tension": tension_loss},
            loss_weights={"theta": self.loss_lambda_theta, "tension": self.loss_lambda_tension},
        )
        self.model_ = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        import tensorflow as tf  # type: ignore

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.shape[1] != (self.theta_dim + self.tension_dim):
            raise ValueError("y dim mismatch for TFDualHeadRegressor")

        y_theta = y[:, : self.theta_dim]
        y_tension = y[:, self.theta_dim :]

        if self.model_ is None:
            self._build_model(input_dim=X.shape[1])

        callbacks: list[Any] = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.n_iter_no_change,
                min_delta=self.tol,
                restore_best_weights=True,
                verbose=0,
            )
        ]

        history = self.model_.fit(
            X,
            {"theta": y_theta, "tension": y_tension},
            epochs=self.max_iter,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
            callbacks=callbacks,
            shuffle=True,
        )

        self.loss_curve_ = [float(v) for v in history.history.get("loss", [])]
        self.validation_scores_ = [float(-v) for v in history.history.get("val_loss", [])]
        self.n_iter_ = int(len(self.loss_curve_))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float32)
        out = self.model_.predict(X, batch_size=self.batch_size, verbose=0)
        theta = np.asarray(out["theta"], dtype=float)
        tension = np.asarray(out["tension"], dtype=float)
        return np.concatenate([theta, tension], axis=1)

