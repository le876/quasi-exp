#!/usr/bin/env python3
"""Hard GPU admission gate for V10 probe-student training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from quasi_exp.teacher.experiment import atomic_write_json


def median_time(fn, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        values.append(time.perf_counter() - started)
    return float(np.median(values))


def run(output: Path, repeats: int) -> dict[str, object]:
    import tensorflow as tf

    nvidia_checks = []
    for _ in range(5):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        nvidia_checks.append({"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()})
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow did not enumerate a GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    rng = np.random.default_rng(20260720)
    x = rng.normal(size=(8192, 128)).astype(np.float32)
    weights = [
        rng.normal(scale=0.05, size=(128, 512)).astype(np.float32),
        rng.normal(scale=0.05, size=(512, 512)).astype(np.float32),
        rng.normal(scale=0.05, size=(512, 6)).astype(np.float32),
    ]

    def execute(device: str) -> np.ndarray:
        with tf.device(device):
            value = tf.convert_to_tensor(x)
            value = tf.nn.gelu(value @ tf.convert_to_tensor(weights[0]))
            value = tf.nn.gelu(value @ tf.convert_to_tensor(weights[1]))
            value = value @ tf.convert_to_tensor(weights[2])
            return value.numpy()

    cpu = execute("/CPU:0")
    gpu = execute("/GPU:0")
    cpu_time = median_time(lambda: execute("/CPU:0"), repeats)
    gpu_time = median_time(lambda: execute("/GPU:0"), repeats)
    max_abs = float(np.max(np.abs(cpu - gpu)))
    speedup = cpu_time / gpu_time
    checks = {
        "nvidia_smi_5_of_5": all(row["returncode"] == 0 for row in nvidia_checks),
        "tensorflow_gpu_enumerated": bool(gpus),
        "actual_gpu_kernel_finite": bool(np.isfinite(gpu).all()),
        "cpu_gpu_parity_atol_1e5": max_abs <= 1.0e-5,
        "speedup_at_least_1p25": speedup >= 1.25,
    }
    report = {
        "protocol_id": "trajectory-v10-gpu-preflight-v1",
        "tensorflow_version": tf.__version__,
        "gpu_devices": [str(device) for device in gpus],
        "nvidia_smi": nvidia_checks,
        "cpu_time_s": cpu_time,
        "gpu_time_s": gpu_time,
        "speedup": speedup,
        "cpu_gpu_max_abs": max_abs,
        "checks": checks,
        "gpu_gate_pass": bool(all(checks.values())),
    }
    atomic_write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.repeats), indent=2, allow_nan=False))
