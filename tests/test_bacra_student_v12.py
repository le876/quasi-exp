from __future__ import annotations

from quasi_exp.teacher.bacra_student import (
    StudentStrategy,
    _packed_targets,
    choose_student_strategy,
    plan_seed_execution,
)
import pandas as pd


def test_strategy_follows_atlas_authorization() -> None:
    assert (
        choose_student_strategy({"static_inverse_authorized": True})
        is StudentStrategy.STATIC
    )
    assert (
        choose_student_strategy({"chart_conditioned_inverse_authorized": True})
        is StudentStrategy.CHART_CONDITIONED
    )
    assert choose_student_strategy({}) is StudentStrategy.STATEFUL


def test_single_gpu_serializes_seeds_and_cpu_parallelizes_them() -> None:
    gpu = plan_seed_execution(
        [1, 2, 3, 4, 5],
        gpu_available=True,
        cpu_workers=5,
        intraop_threads=2,
        interop_threads=1,
    )
    assert gpu.device == "gpu:0"
    assert gpu.effective_workers == 1
    cpu = plan_seed_execution(
        [1, 2, 3, 4, 5],
        gpu_available=False,
        cpu_workers=5,
        intraop_threads=2,
        interop_threads=1,
    )
    assert cpu.device == "cpu"
    assert cpu.effective_workers == 5


def test_bacra_packed_targets_use_regional_schema() -> None:
    frame = pd.DataFrame(
        [
            {
                "x_m": 1.0,
                "y_m": 2.0,
                "z_m": 3.0,
                **{f"beta{index}_rad": index / 10.0 for index in range(1, 7)},
            }
        ]
    )
    packed = _packed_targets(frame)
    assert packed.shape == (1, 9)
    assert packed[0, 6:].tolist() == [1.0, 2.0, 3.0]
