from __future__ import annotations

import subprocess
from typing import Any


_GPU_DEVICE_VALUES = {"gpu", "cuda"}
_GPU_RUNTIME_ERROR_PATTERNS = (
    "no opencl device found",
    "opencl",
    "gpu tree learner was not enabled",
    "no gpu device found",
    "cannot use gpu",
    "cuda error",
    "cl_device_not_found",
    "cl_platform_not_found",
)
_GPU_PROBE_CACHE: tuple[bool, str] | None = None


def _is_gpu_value(value: Any) -> bool:
    return str(value).strip().lower() in _GPU_DEVICE_VALUES


def is_lgbm_gpu_requested(params: dict[str, Any] | None) -> bool:
    if not isinstance(params, dict):
        return False

    for prefix in ("", "estimator__"):
        for key in ("device", "device_type"):
            full_key = f"{prefix}{key}"
            if full_key in params and _is_gpu_value(params.get(full_key)):
                return True
    return False


def force_lgbm_params_to_cpu(params: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(params or {})
    for prefix in ("", "estimator__"):
        for key in ("device", "device_type"):
            full_key = f"{prefix}{key}"
            if full_key in out and _is_gpu_value(out.get(full_key)):
                out[full_key] = "cpu"

        for key in ("gpu_use_dp", "gpu_platform_id", "gpu_device_id"):
            out.pop(f"{prefix}{key}", None)
    return out


def is_lgbm_gpu_runtime_error(exc: BaseException) -> bool:
    msg = str(exc).strip().lower()
    if not msg:
        return False
    return any(pattern in msg for pattern in _GPU_RUNTIME_ERROR_PATTERNS)


def probe_lgbm_gpu_available(timeout_s: float = 5.0) -> tuple[bool, str]:
    global _GPU_PROBE_CACHE
    if _GPU_PROBE_CACHE is not None:
        return _GPU_PROBE_CACHE

    notes: list[str] = []
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            if err:
                notes.append(f"nvidia-smi failed: {err}")
            else:
                notes.append(f"nvidia-smi failed with return code {proc.returncode}")
    except Exception as exc:  # pragma: no cover - system dependent
        notes.append(f"nvidia-smi check failed: {exc}")

    try:
        import numpy as np
        from lightgbm import LGBMRegressor

        X = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.2, -0.1], [1.0, 0.0, 0.1], [0.2, -0.3, 0.7]], dtype=np.float32)
        y = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        probe = LGBMRegressor(
            device="gpu",
            n_estimators=1,
            learning_rate=0.3,
            num_leaves=4,
            min_data_in_leaf=1,
            max_depth=2,
            verbosity=-1,
        )
        probe.fit(X, y)
    except Exception as exc:
        notes.append(f"lightgbm gpu probe failed: {exc}")
        _GPU_PROBE_CACHE = (False, "; ".join(notes))
        return _GPU_PROBE_CACHE

    if notes:
        _GPU_PROBE_CACHE = (True, "; ".join(notes))
    else:
        _GPU_PROBE_CACHE = (True, "ok")
    return _GPU_PROBE_CACHE
