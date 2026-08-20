from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from quasi_exp.teacher.retry8_progression import (
    authorize_five_k_after_smoke,
    map_frozen_lineage_roots,
    select_balanced_smoke_domain,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_3_repaired_5k_student.py"


def _runner():
    spec = importlib.util.spec_from_file_location("v143_retry8", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry8_smoke_config_is_small_and_keeps_formal_thresholds() -> None:
    config = _runner().load_config(
        ROOT / "configs/bacra_v14_3_retry8_smoke.yaml"
    )

    assert config["upstream_protocol_version"] == "retry8"
    assert config["execution_mode"] == "smoke"
    assert config["pilot"]["parent_cell_count"] == 1000
    assert config["pilot"]["task_probe_count"] == 5000
    assert config["pilot"]["root_count"] == 24
    assert config["dataset"]["minimum_unique_rows"] == 3000
    assert config["student"]["seeds"] == [20260881]
    assert config["formal_gate"]["labelable_measure_min"] == 0.80

    five_k = _runner().load_config(ROOT / "configs/bacra_v14_3_retry8_5k.yaml")
    assert five_k["execution_mode"] == "five_k"
    assert five_k["pilot"]["parent_cell_count"] == 5000
    assert five_k["pilot"]["minimum_x_tertiles"] == 2
    assert five_k["dataset"]["minimum_unique_rows"] == 3000
    assert five_k["dataset"]["target_unique_rows"] == 10000


def test_balanced_smoke_keeps_p0_and_only_a_registered_hard_subset() -> None:
    parents = pd.DataFrame(
        {
            "node_id": np.arange(1200),
            "x_m": np.arange(1200) / 1000.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "strata": ["interior"] * 1200,
        }
    )
    tasks = pd.DataFrame(
        {
            "task_node_id": np.arange(1200),
            "source_parent_node_id": np.arange(1200),
            "core_safe": [True] * 1200,
            "x_m": np.arange(1200) / 1000.0,
            "y_m": 0.0,
            "z_m": 0.0,
        }
    )
    frozen = pd.DataFrame(
        {
            "source_parent_node_id": np.arange(20),
            "x_m": np.arange(20) / 1000.0,
            "y_m": 0.0,
            "z_m": 0.0,
        }
    )

    selected = select_balanced_smoke_domain(
        parents,
        tasks,
        frozen,
        hard_parent_ids=set(range(300)),
        parent_count=1000,
        hard_parent_target=192,
        hard_parent_maximum=256,
    )

    assert len(selected) == 1000
    assert set(range(20)).issubset(set(selected))
    assert 192 <= len(set(selected) & set(range(300))) <= 256


def test_lineage_roots_copy_beta_at_distinct_spatial_nodes() -> None:
    tasks = pd.DataFrame(
        {
            "task_node_id": [10, 11, 12, 13],
            "x_m": [0.0, 0.02, 0.04, 0.06],
            "y_m": 0.0,
            "z_m": 0.0,
        }
    )
    labels = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "canonical_lineage_id": ["lineage"] * 4,
            "x_m": [0.0, 0.02, 0.04, 0.06],
            "y_m": 0.0,
            "z_m": 0.0,
            **{f"beta{index}_rad": [0.0, 0.1, 0.2, 0.3] for index in range(1, 7)},
        }
    )

    roots = map_frozen_lineage_roots(labels, tasks, root_count=3)

    assert len(roots) == 3
    assert roots["task_node_id"].nunique() == 3
    assert roots["canonical_lineage_id"].nunique() == 1
    assert roots["root_kind"].tolist()[0] == "branch_origin"
    assert set(roots["root_kind"].tolist()[1:]) == {"spatial_relay"}


def test_student_quality_miss_does_not_block_five_k_teacher() -> None:
    quality_miss = authorize_five_k_after_smoke(
        smoke_pipeline_operationally_complete=True,
        teacher_label_integrity_failure=False,
        unresolved_training_implementation_failure=False,
        smoke_student_quality_pass=False,
    )
    integrity_failure = authorize_five_k_after_smoke(
        smoke_pipeline_operationally_complete=True,
        teacher_label_integrity_failure=True,
        unresolved_training_implementation_failure=False,
        smoke_student_quality_pass=True,
    )

    assert quality_miss["five_k_teacher_execution_authorized"] is True
    assert quality_miss["student_claim_authorized"] is False
    assert integrity_failure["five_k_teacher_execution_authorized"] is False
