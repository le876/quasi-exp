from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))

import run_bacra_v12 as v12
import run_bacra_v12_16b_chart_conditioned_student as runner
from quasi_exp.teacher.multichart_distillation import (
    build_chart_conditioned_model,
    predict_chart_conditioned,
)


PYTHON = Path(
    "/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python"
)
SOURCE_MODEL = Path(
    "/mnt/ML_projects/quasi_exp/runs/"
    "bacra_v12_15_single_student_formal_margin_tail_retry1_20260730/"
    "05_model_lock/seed_20260738/model.keras"
)


def test_config_freezes_known_chart_scope() -> None:
    config = v12.load_protocol_config(
        ROOT / "configs/bacra_v12_16b_chart_conditioned_student.yaml",
        "smoke",
    )
    assert config["claim_scope"] == runner.CLAIM_SCOPE
    assert config["deployment_claim_gate_pass"] is False
    assert config["v12_16b"]["parallel"]["gpu_training_workers"] == 1
    assert config["v12_16b"]["parallel"]["cpu_workers"] == 4


def test_known_chart_model_exactly_preserves_chart_a_initialization() -> None:
    import tensorflow as tf

    source = tf.keras.models.load_model(SOURCE_MODEL, compile=False)
    model = build_chart_conditioned_model(source)
    xyz = np.asarray(
        [
            [0.10, 0.02, 0.31],
            [0.11, -0.01, 0.30],
            [0.09, 0.00, 0.32],
        ],
        dtype=np.float32,
    )
    expected = source.predict(xyz, verbose=0)
    actual = predict_chart_conditioned(model, xyz, 0.0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)
    assert len(model.inputs) == 2


def test_point_gate_uses_registered_thresholds() -> None:
    config = v12.load_protocol_config(
        ROOT / "configs/bacra_v12_16b_chart_conditioned_student.yaml",
        "formal",
    )
    frame = pd.DataFrame(
        [
            {
                "fk_p95_mm": 4.9,
                "fk_max_mm": 9.9,
                "minimum_joint_margin_deg": 1.51,
                "actual_bounds": True,
            },
            {
                "fk_p95_mm": 5.1,
                "fk_max_mm": 9.0,
                "minimum_joint_margin_deg": 1.6,
                "actual_bounds": True,
            },
        ]
    )
    gated = runner._apply_point_gate(frame, config)
    assert gated["seed_gate_pass"].tolist() == [True, False]
