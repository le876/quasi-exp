from __future__ import annotations

import importlib.util
import json
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


def test_retry8_pipelines_seal_reach_and_formal_before_summary() -> None:
    for name in (
        "run_bacra_v14_3_retry8_smoke.sh",
        "run_bacra_v14_3_retry8_5k.sh",
    ):
        script = (ROOT / "scripts/pipelines" / name).read_text(encoding="utf-8")
        stages = [
            line.removeprefix("run_stage ").strip()
            for line in script.splitlines()
            if line.startswith("run_stage ")
        ]

        assert stages[-4:] == [
            "representation_decision",
            "reach_update",
            "formal_admission",
            "summary",
        ]


def test_retry8_reach_update_reuses_registered_retry7_round8(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = _runner()
    retry7_root = tmp_path / "retry7"
    reach_gate = retry7_root / "07_reach_round8/gate.json"
    reach_gate.parent.mkdir(parents=True)
    reach_gate.write_text(
        json.dumps({"gate_pass": False, "reach_convergence_gate": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "_sources",
        lambda *_: {"v14_2r": tmp_path / "retry8", "retry7": retry7_root},
    )
    monkeypatch.setattr(runner, "_require", lambda *_args, **_kwargs: {})

    report = runner.stage_reach_update(
        {"upstream_protocol_version": "retry8"},
        tmp_path,
        tmp_path / "output",
    )

    assert report["gate_pass"] is False
    assert report["reach_source"] == "retry7_round8"
    assert report["round8_executed_upstream"] is True
    assert report["round9_forbidden"] is True


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


def test_retry8_inherited_candidates_keep_integer_local_cluster_ids(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = _runner()
    tasks = pd.DataFrame(
        {
            "task_node_id": [10],
            "x_m": [1.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "source_parent_node_id": [1],
        }
    )
    candidates = pd.DataFrame.from_records(
        [
            {
                "task_node_id": 10,
                "candidate_id": "legacy_10",
                "cluster_id": 0,
                "source_candidate_ids": ["legacy_10"],
                "cluster_size": 1,
                "quality": "Gold",
                "solver_success": True,
                "actual_bounds": True,
                "residual_mm": 0.0,
                "min_margin_deg": 1.0,
                "normalized_min_margin": 0.1,
                "posture_cost": 0.0,
                "condition_number": 1.0,
                **{column: 0.0 for column in runner.BETA_COLUMNS},
            }
        ]
    )
    frozen = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "frozen_10",
                "canonical_lineage_id": "lineage",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 0.0,
                **{column: 0.1 for column in runner.BETA_COLUMNS},
            }
        ]
    )
    lineage_path = tmp_path / "lineage.parquet"
    frozen.to_parquet(lineage_path, index=False)
    monkeypatch.setattr(
        runner,
        "_legacy_pilot_inputs",
        lambda *_: (tasks, pd.DataFrame(), candidates, pd.DataFrame()),
    )
    monkeypatch.setattr(
        runner,
        "_upstream_stage_paths",
        lambda *_: {"lineage": lineage_path},
    )
    monkeypatch.setattr(runner, "_sources", lambda *_: {"v14_2r": tmp_path})

    _tasks, _edges, merged, _parents, _roots = runner._retry8_pilot_inputs(
        {
            "execution_mode": "five_k",
            "pilot": {"root_count": 1},
        },
        tmp_path,
    )
    output = tmp_path / "candidate_clusters.parquet"
    runner._write_parquet(merged, output)

    restored = pd.read_parquet(output)
    assert restored["cluster_id"].map(type).eq(int).all()
    assert sorted(restored["cluster_id"].tolist()) == [0, 1]
