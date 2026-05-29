from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from lgbm_device_fallback import force_lgbm_params_to_cpu, is_lgbm_gpu_runtime_error


def test_force_lgbm_params_to_cpu_handles_direct_gpu_keys() -> None:
    params = {
        "device": "gpu",
        "gpu_use_dp": True,
        "gpu_platform_id": 0,
        "gpu_device_id": 1,
        "n_estimators": 300,
    }

    out = force_lgbm_params_to_cpu(params)
    assert out["device"] == "cpu"
    assert "gpu_use_dp" not in out
    assert "gpu_platform_id" not in out
    assert "gpu_device_id" not in out
    assert out["n_estimators"] == 300


def test_force_lgbm_params_to_cpu_handles_estimator_prefixed_keys() -> None:
    params = {
        "estimator__device": "gpu",
        "estimator__gpu_use_dp": True,
        "estimator__gpu_platform_id": 0,
        "estimator__gpu_device_id": 1,
        "estimator__n_estimators": 400,
    }

    out = force_lgbm_params_to_cpu(params)
    assert out["estimator__device"] == "cpu"
    assert "estimator__gpu_use_dp" not in out
    assert "estimator__gpu_platform_id" not in out
    assert "estimator__gpu_device_id" not in out
    assert out["estimator__n_estimators"] == 400


def test_is_lgbm_gpu_runtime_error_matches_opencl_failure() -> None:
    exc = RuntimeError("No OpenCL device found")
    assert is_lgbm_gpu_runtime_error(exc)


def test_is_lgbm_gpu_runtime_error_ignores_non_gpu_error() -> None:
    exc = RuntimeError("Input data has NaN")
    assert not is_lgbm_gpu_runtime_error(exc)
