#!/usr/bin/env python3
"""Train the registered S0/S2/S4 Pilot-C representations on one teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from quasi_exp.teacher.experiment import atomic_write_json
from quasi_exp.teacher.student import StudentKind, build_student_arrays, split_by_trajectory


def model_for(tf, input_dim: int, output_dim: int):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input((input_dim,)),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(128, activation="gelu"),
            tf.keras.layers.Dense(128, activation="gelu"),
            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.Dense(output_dim),
        ]
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(args.seed))
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Pilot C requires a TensorFlow GPU after gpu_preflight_v10")
    frame = pd.read_parquet(args.dataset)
    splits = split_by_trajectory(frame, seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for kind_name in args.students:
        kind = StudentKind(kind_name)
        arrays = {role: build_student_arrays(split, kind) for role, split in splits.items()}
        if kind == StudentKind.S4:
            # Pilot S4 uses one joint classifier+expert network; expert beta
            # loss remains explicit and chart accuracy remains separately auditable.
            chart_count = int(frame["chart_id"].max()) + 1
            inputs = tf.keras.Input((3,))
            norm = tf.keras.layers.Normalization()
            norm.adapt(arrays["train"].features)
            hidden = tf.keras.layers.Dense(128, activation="gelu")(norm(inputs))
            hidden = tf.keras.layers.Dense(128, activation="gelu")(hidden)
            chart_out = tf.keras.layers.Dense(chart_count, name="chart")(hidden)
            expert_outputs = [
                tf.keras.layers.Dense(6, name=f"expert_{chart_index}")(hidden)
                for chart_index in range(chart_count)
            ]
            experts = tf.keras.layers.Lambda(lambda values: tf.stack(values, axis=1), name="experts")(
                expert_outputs
            )
            probabilities = tf.keras.layers.Softmax(name="chart_probabilities")(chart_out)
            beta_out = tf.keras.layers.Lambda(
                lambda values: tf.reduce_sum(values[0] * values[1][..., None], axis=1),
                name="beta",
            )([experts, probabilities])
            model = tf.keras.Model(inputs, [beta_out, chart_out])
            model.compile(
                optimizer="adam",
                loss={"beta": "mse", "chart": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
                loss_weights={"beta": 1.0, "chart": 0.1},
            )
            train_y = {"beta": arrays["train"].targets, "chart": arrays["train"].chart_labels}
            val_y = {"beta": arrays["validation"].targets, "chart": arrays["validation"].chart_labels}
        else:
            model = model_for(tf, arrays["train"].features.shape[1], 6)
            model.layers[0].adapt(arrays["train"].features)
            model.compile(optimizer="adam", loss="mse")
            train_y = arrays["train"].targets
            val_y = arrays["validation"].targets
        started = time.perf_counter()
        history = model.fit(
            arrays["train"].features,
            train_y,
            validation_data=(arrays["validation"].features, val_y),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=0,
        )
        elapsed = time.perf_counter() - started
        prediction = model.predict(arrays["test"].features, verbose=0)
        beta_prediction = prediction[0] if kind == StudentKind.S4 else prediction
        if kind == StudentKind.S2:
            previous = splits["test"].loc[:, [f"previous_beta{i}_rad" for i in range(1, 7)]].to_numpy()
            beta_prediction = beta_prediction + previous
            beta_target = arrays["test"].targets + previous
        else:
            beta_target = arrays["test"].targets
        beta_error = np.rad2deg(np.sqrt(np.mean(np.square(beta_prediction - beta_target), axis=1)))
        kind_dir = output / kind.value
        kind_dir.mkdir(exist_ok=True)
        model.save(kind_dir / "model.keras")
        result = {
            "student": kind.value,
            "epochs_ran": len(history.history["loss"]),
            "wall_time_s": elapsed,
            "test_rows": len(beta_error),
            "beta_rms_p95_deg": float(np.percentile(beta_error, 95)),
            "beta_rms_max_deg": float(np.max(beta_error)),
        }
        atomic_write_json(kind_dir / "report.json", result)
        results.append(result)
    report = {"students": results, "gpu_devices": [str(value) for value in tf.config.list_physical_devices("GPU")]}
    atomic_write_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--students", nargs="+", choices=("S0", "S2", "S4"), default=("S0", "S2", "S4"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))
