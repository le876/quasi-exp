from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_3r_retry10.py"
CONFIG = ROOT / "configs/bacra_v14_3r_retry10_frontier_expanded_atlas.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("bacra_v14_3r_retry10", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry10_stage_dag_and_claim_boundaries_are_frozen() -> None:
    module = _module()
    config = module.load_config(CONFIG)
    assert module.STAGE_ORDER == (
        "inventory",
        "atlas_poc",
        "seed_datasets",
        "seed_students",
        "atlas_expansion",
        "taxonomy_rescue",
        "atlas_freeze",
        "full_diagnostic",
        "preflight_50k",
        "dataset_50k",
        "student_50k",
        "tension_pilot",
        "preflight_100k",
        "dataset_100k",
        "student_100k",
        "preflight_200k",
        "dataset_200k",
        "student_200k",
        "tension_materialization",
        "summary",
    )
    assert config["metric"]["version"] == "normalized_weighted_v1"
    assert config["atlas"]["root_count_per_batch"] == 4
    assert config["atlas"]["maximum_batches"] == 6
    assert config["sampling"]["preflight_rows"] == 2000
    assert config["claims"]["formal_authorized"] is False
    assert config["claims"]["deployment_authorized"] is False
    assert config["claims"]["full_workspace_authorized"] is False


def test_dataset_and_student_authorizations_are_separate_in_source() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    dataset_function = source.split("def _dataset_stage(", 1)[1].split(
        "def stage_dataset_50k", 1
    )[0]
    assert "student_quality" not in dataset_function
    assert "tension_pilot" not in dataset_function
    assert "formal_authorized" in dataset_function
    assert "row_padding_used" in dataset_function
    assert '"student_scaling_claim_authorized": False' in source


def test_preflight_is_not_dataset_completion() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    preflight = source.split("def _preflight_stage(", 1)[1].split(
        "def stage_preflight_50k", 1
    )[0]
    assert "preflight_generation_decision" in preflight
    assert '"dataset_complete":' not in preflight
    assert "preflight_attempts.parquet" in preflight


def test_launcher_forces_cpu_before_any_runner_process() -> None:
    launcher = (
        ROOT / "scripts/pipelines/run_bacra_v14_3r_retry10.sh"
    ).read_text(encoding="utf-8")
    assert "export CUDA_VISIBLE_DEVICES=-1" in launcher
    assert launcher.index("export CUDA_VISIBLE_DEVICES=-1") < launcher.index('"$PYTHON_BIN" "$RUNNER"')
    assert "--validate-stage" in launcher


def test_protocol_removes_unregistered_cell_size_and_full_workspace_name() -> None:
    protocol = (
        ROOT / "docs/27-BACRA-V14.3R-retry10-frontier-expanded-atlas执行协议.md"
    ).read_text(encoding="utf-8")
    assert "禁止 hardcode `10 mm`" in protocol
    assert "full_workspace_dataset" in protocol
    assert "只有未来独立" in protocol
    assert "pilot_domain_expanded_exploratory_dataset_200k" not in protocol or "pilot_domain" in protocol


def test_tension_replacement_scope_is_registered_and_not_global() -> None:
    module = _module()
    config = module.load_config(CONFIG)
    assert config["tension"]["replacement_scope"] == [
        "source_parent_node_id",
        "macroblock_id",
        "sampling_stratum",
    ]
    source = RUNNER.read_text(encoding="utf-8")
    assert "same_stratum_backup" in source
    assert "replacement_key" in source


def test_dataset_integrity_rejects_cross_scale_point_id_collisions() -> None:
    module = _module()
    rows = []
    for index in range(2):
        rows.append(
            {
                "xyz_key": f"xyz-{index}",
                "physical_point_id": "duplicated-id",
                "actual_bounds": True,
                "x_m": float(index),
                "y_m": 0.0,
                "z_m": 0.0,
                **{f"beta{axis}_rad": 0.0 for axis in range(1, 7)},
            }
        )
    integrity = module._dataset_integrity(pd.DataFrame.from_records(rows), 2)
    assert integrity["unique_xyz_pass"] is True
    assert integrity["physical_point_id_unique_pass"] is False
    assert module._stable_point_id("xyz-0") != module._stable_point_id("xyz-1")


def test_atlas_candidate_repairs_only_nonfinite_optional_metadata() -> None:
    module = _module()
    beta = np.linspace(-0.25, 0.25, 6)
    row = SimpleNamespace(
        task_node_id=17,
        candidate_id="poc-root-17",
        source_teacher_residual_mm=np.nan,
        residual_mm=0.012,
        min_margin_deg=np.nan,
        normalized_min_margin=np.inf,
        posture_cost=np.nan,
        condition_number=np.inf,
        quality=None,
        label_quality="Silver",
        **{name: float(value) for name, value in zip(module.BETA_COLUMNS, beta, strict=True)},
    )

    candidate = module._atlas_candidate(row)

    assert np.array_equal(candidate.beta_rad, beta)
    assert candidate.residual_mm == 0.012
    assert candidate.min_margin_deg == 1.0
    assert candidate.normalized_min_margin == 1.0 / 180.0
    assert candidate.posture_cost == np.linalg.norm(beta)
    assert candidate.condition_number == 1.0
    assert candidate.quality == "Silver"


def test_growth_merge_metadata_is_persisted_as_finite() -> None:
    module = _module()
    beta = np.arange(12, dtype=float).reshape(2, 6) / 20.0
    frame = pd.DataFrame(
        {
            **{name: beta[:, index] for index, name in enumerate(module.BETA_COLUMNS)},
            "residual_mm": [0.01, np.nan],
            "min_margin_deg": [2.0, np.nan],
            "normalized_min_margin": [2.0 / 180.0, np.nan],
            "posture_cost": [np.nan, np.inf],
            "condition_number": [3.0, np.inf],
        }
    )

    normalized = module._normalize_candidate_metadata(frame)

    for name in (
        "residual_mm",
        "min_margin_deg",
        "normalized_min_margin",
        "posture_cost",
        "condition_number",
    ):
        assert np.isfinite(normalized[name].to_numpy(float)).all()
    assert normalized["posture_cost"].tolist() == np.linalg.norm(beta, axis=1).tolist()
    assert normalized.loc[1, "condition_number"] == 1.0


def test_targeted_rescue_can_relay_from_a_newly_rescued_node(monkeypatch) -> None:
    module = _module()
    beta = np.zeros(6, dtype=float)
    labels = pd.DataFrame.from_records(
        [
            {
                "task_node_id": 1,
                "candidate_id": "seed-1",
                "quality": "Gold",
                "actual_bounds": True,
                **{name: 0.0 for name in module.XYZ_COLUMNS},
                **{name: float(value) for name, value in zip(module.BETA_COLUMNS, beta, strict=True)},
            }
        ]
    )
    tasks = pd.DataFrame.from_records(
        [
            {
                "task_node_id": node_id,
                "source_parent_node_id": node_id,
                "x_m": float(node_id - 1) / 1000.0,
                "y_m": 0.0,
                "z_m": 0.0,
            }
            for node_id in (1, 2, 3)
        ]
    )
    edges = pd.DataFrame.from_records(
        [
            {"left_node_id": 1, "right_node_id": 2},
            {"left_node_id": 2, "right_node_id": 3},
        ]
    )
    taxonomy = pd.DataFrame.from_records(
        [
            {"task_node_id": 2, "taxonomy": "frontier_reachable", "taxonomy_priority": 1},
            {"task_node_id": 3, "taxonomy": "frontier_reachable", "taxonomy_priority": 1},
        ]
    )

    def continuation(source, target):
        return SimpleNamespace(
            success=True,
            actual_bounds=True,
            residual_mm=0.01,
            beta_rad=source.beta_rad.copy(),
        )

    monkeypatch.setattr(module, "_environment", lambda _config: object())
    monkeypatch.setattr(
        module,
        "make_optimized_predictor_corrector_continuation",
        lambda *_args, **_kwargs: continuation,
    )
    monkeypatch.setattr(module, "build_shared_face_task_edges", lambda _tasks, active: active)
    config = {
        "atlas": {
            "individual_fk_max_mm": 0.05,
            "weighted_local_p95_max_deg": 2.0,
            "raw_catastrophic_deg": 5.0,
        }
    }

    merged, rescued = module._targeted_rescue(config, labels, tasks, edges, taxonomy)

    assert rescued["task_node_id"].tolist() == [2, 3]
    assert set(merged["task_node_id"].astype(int)) == {1, 2, 3}
    assert np.isfinite(
        rescued[
            [
                "residual_mm",
                "teacher_fk_residual_mm",
                "min_margin_deg",
                "normalized_min_margin",
                "posture_cost",
                "condition_number",
            ]
        ].to_numpy(float)
    ).all()
