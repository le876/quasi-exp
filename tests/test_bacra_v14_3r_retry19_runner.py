from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import scripts.analysis.run_bacra_v14_3r_retry19_fullspace_direct_student as retry19
from scripts.analysis.run_bacra_v14_3r_retry19_fullspace_direct_student import (
    EXPERIMENT_ID,
    STAGE_DIRS,
    _coverage_policy,
    _teacher_policy,
    _write_objective_feasibility_contract,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bacra_v14_3r_retry19_fullspace_direct_student.yaml"


def test_retry19_config_locks_budget_domains_and_mixed_coverage_gates() -> None:
    config = load_config(CONFIG)
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["workspace"]["primary_u_maximum_mm"] == 200.0
    assert config["workspace"]["audit_u_maximum_mm"] == 220.0
    assert config["workspace"]["primary_target_cap"] == 30000
    assert config["workspace"]["audit_target_cap"] == 512
    assert config["workspace"]["pilot_target_count"] == 5000
    assert config["workspace"]["immutable_batch_size"] == 1000
    assert config["workspace"]["fine_root_radius_mm"] == 40.0
    policy = _coverage_policy(config)
    assert policy.volume_coverage_minimum == 0.90
    assert policy.fine_coverage_minimum == 0.85
    assert policy.coarse_coverage_minimum == 0.90
    assert policy.maximum_hole_fraction == 0.02
    assert policy.minimum_axial_slice_coverage == 0.80
    assert policy.path_success_minimum == 0.95
    assert policy.stretch_p95_maximum == 2.0


def test_retry19_graph_teacher_and_student_contract_are_frozen() -> None:
    config = load_config(CONFIG)
    policy = _teacher_policy(config)
    assert policy.k == 16
    assert policy.sigma_mm == 10.0
    assert policy.pairwise_lambda == 4.0
    assert policy.huber_delta_deg == 7.0
    assert policy.soft_anchor_bonus == 0.25
    assert policy.maximum_icm_sweeps == 12
    assert config["graph_teacher"]["pairwise_lambdas"] == [4.0]
    assert config["student"]["signed_power_epsilon_mm"] == 3.0
    assert config["student"]["edge_minimum_distance_mm"] == 5.0
    assert config["student"]["edge_registry_cap"] == 65536
    assert config["student"]["maximum_steps"] == 500
    assert config["runtime"]["trajectory_candidate_workers"] == 12


def test_retry19_exact_zero_has_one_legal_hard_anchor_candidate(tmp_path: Path) -> None:
    config = retry19.load_config(CONFIG)
    targets = pd.DataFrame(
        [
            {
                "target_id": "retry19_exact_zero",
                "x_m": 1.215498,
                "y_m": 0.0,
                "z_m": 0.0,
                "target_role": "exact_zero",
                "domain_class": "primary",
                "old_label_candidate": True,
                **{column: 0.0 for column in retry19.BETA_COLUMNS},
                "fk_residual_mm": float("nan"),
                "selection_ordinal": 0,
            }
        ]
    )
    anchors = retry19._anchor_registry(targets)
    fresh = retry19._solve_target_batch(
        config,
        targets,
        tmp_path,
        smoke=False,
        namespace="exact_zero_regression",
    )
    candidates = pd.concat(
        [fresh, retry19._candidate_rows_for_old(targets, anchors)],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(["target_id", "candidate_id"], keep="last")

    labels, _edges, trials = retry19._select_teacher(config, candidates, targets, anchors)

    assert labels["candidate_id"].tolist() == ["retry19_exact_zero_candidate"]
    assert labels["old_label_candidate"].astype(bool).tolist() == [True]
    assert trials["status"].eq("complete").all()


def test_graph_teacher_materializes_finite_target_coordinates_before_cell_service(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(CONFIG)
    target_id = "target"
    target = pd.DataFrame(
        [
            {
                "target_id": target_id,
                "x_m": 1.215,
                "y_m": 0.0,
                "z_m": 0.0,
                "domain_class": "primary",
                "old_label_candidate": False,
            }
        ]
    )
    candidate = pd.DataFrame(
        [
            {
                "target_id": target_id,
                "candidate_id": "candidate",
                "x_m": math.nan,
                "y_m": math.nan,
                "z_m": math.nan,
                "fk_residual_mm": 0.0,
                **{column: 0.0 for column in retry19.BETA_COLUMNS},
            }
        ]
    )
    audit_candidate = candidate.assign(
        target_id="audit-target",
        candidate_id="audit-candidate",
    )
    cells = pd.DataFrame(
        [
            {
                "cell_id": "cell",
                "domain_class": "primary",
                "volume_mm3": 125.0,
                "required": True,
                "geometric_zero_connected": True,
            }
        ]
    )
    empty_edges = pd.DataFrame(columns=["left_target_id", "right_target_id"])
    discovery = tmp_path / STAGE_DIRS["fullspace_discovery"]
    adaptive = tmp_path / STAGE_DIRS["adaptive_fill"]
    discovery.mkdir(parents=True)
    adaptive.mkdir(parents=True)
    target.to_parquet(adaptive / "final_primary_target_registry.parquet", index=False)
    pd.concat([candidate, audit_candidate], ignore_index=True).to_parquet(
        adaptive / "full_candidate_bank.parquet",
        index=False,
    )
    cells.to_parquet(discovery / "mixed_resolution_cell_registry.parquet", index=False)
    empty_edges.to_parquet(discovery / "primary_face_adjacency.parquet", index=False)
    pd.DataFrame().to_parquet(discovery / "path_query_registry.parquet", index=False)

    monkeypatch.setattr(retry19, "_gate", lambda _root, stage: {"status": "complete"} if stage == "adaptive_fill" else {})
    monkeypatch.setattr(retry19, "_anchor_registry", lambda _targets: pd.DataFrame())
    monkeypatch.setattr(retry19, "_candidate_rows_for_old", lambda _targets, _anchors: pd.DataFrame(columns=candidate.columns))
    def select_primary_only(_config, candidates, targets, _anchors):
        assert set(candidates["target_id"]) == set(targets["target_id"]) == {target_id}
        return candidate.copy(), empty_edges.copy(), pd.DataFrame()

    monkeypatch.setattr(retry19, "_select_teacher", select_primary_only)
    monkeypatch.setattr(retry19, "teacher_zero_attachment", lambda *_args, **_kwargs: (True, {target_id}, pd.DataFrame()))

    def assert_finite_coordinates(frame, labels, **_kwargs):
        assert np.isfinite(labels.loc[:, retry19.XYZ_COLUMNS].to_numpy(float)).all()
        return frame.assign(served=True, service_distance_mm=0.0)

    monkeypatch.setattr(retry19, "mark_served_cells", assert_finite_coordinates)
    monkeypatch.setattr(retry19, "mixed_resolution_coverage_metrics", lambda _frame: {})
    monkeypatch.setattr(retry19, "largest_unserved_component_fraction", lambda *_args: 0.0)
    monkeypatch.setattr(retry19, "axial_slice_coverage", lambda *_args, **_kwargs: pd.DataFrame({"volume_coverage": [1.0]}))
    monkeypatch.setattr(retry19, "data_gate", lambda *_args, **_kwargs: {"checks": {}})
    monkeypatch.setattr(retry19, "teacher_edge_metrics", lambda *_args, **_kwargs: {"weighted_p95_deg": 0.0, "raw_gt7_rate": 0.0})

    gate = retry19.stage_graph_teacher(config, tmp_path, smoke=False)

    assert gate["status"] == "green"


def test_materialize_jacobians_uses_vectorized_forward_contract() -> None:
    frame = pd.DataFrame(
        [
            {column: float(index) for index, column in enumerate(retry19.BETA_COLUMNS)},
            {column: float(index + 10) for index, column in enumerate(retry19.BETA_COLUMNS)},
        ]
    )

    class BatchOnlyEnvironment:
        def fk_and_jacobian(self, beta):
            assert np.asarray(beta).shape == (2, 6)
            return np.zeros((2, 3)), np.arange(36, dtype=float).reshape(2, 3, 6)

        def jacobian(self, _beta):
            raise AssertionError("scalar jacobian API must not receive a batch")

    result = retry19._materialize_jacobians(frame, BatchOnlyEnvironment())

    assert result.loc[:, retry19.JACOBIAN_COLUMNS].to_numpy(float).shape == (2, 18)
    assert result.loc[:, retry19.JACOBIAN_COLUMNS].to_numpy(float).ravel().tolist() == list(
        np.arange(36, dtype=float)
    )


def test_retry19_stage_order_matches_authorization_and_lock_dag() -> None:
    names = list(STAGE_DIRS)
    assert names == [
        "fullspace_discovery",
        "causal_controls",
        "candidate_pilot",
        "continuation_authorization",
        "adaptive_fill",
        "graph_teacher",
        "unified_dataset",
        "student_ablations",
        "student_lock",
        "postlock_trajectories",
        "trajectory_teacher",
        "trajectory_evaluation",
        "summary",
    ]
    assert list(STAGE_DIRS.values())[0].startswith("00_")
    assert list(STAGE_DIRS.values())[-1].startswith("12_")


def test_exploratory_continuation_does_not_turn_diagnostic_red_into_a_block(monkeypatch, tmp_path: Path) -> None:
    gates = {
        "fullspace_discovery": {
            "objective_resource_feasible": False,
            "geometric_zero_connected": True,
        },
        "candidate_pilot": {
            "accepted_target_count": 1,
            "acceptance_rate": 0.10,
        },
    }
    monkeypatch.setattr(retry19, "_gate", lambda _root, stage: gates[stage])
    monkeypatch.setattr(retry19, "_write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        retry19,
        "_seal_gate",
        lambda _root, _config, _stage, gate: dict(gate),
    )

    gate = retry19.stage_continuation_authorization({}, tmp_path, smoke=False)

    assert gate["continuation_authorized"] is True
    assert "OBJECTIVE_RESOURCE_INFEASIBLE" in gate["reason_codes"]
    assert "PILOT_CANDIDATE_ACCEPTANCE_RED" in gate["reason_codes"]


def test_retry19_protocol_launcher_and_config_bind_one_experiment() -> None:
    config = load_config(CONFIG)
    assert (ROOT / config["sources"]["governing_protocol"]).is_file()
    launcher = (ROOT / "scripts/pipelines/run_bacra_v14_3r_retry19_fullspace_direct_student.sh").read_text(encoding="utf-8")
    assert "BACRA_RETRY19_FULLSPACE_DIRECT_BINDING_SHA" in launcher
    assert "run_bacra_v14_3r_retry19_fullspace_direct_student.py" in launcher
    assert "CUDA_VISIBLE_DEVICES=-1" in launcher
    assert config["claims"]["diagnostic_only"]
    assert config["claims"]["exploratory_e2e_authorized"]
    assert not any(value for key, value in config["claims"].items() if key not in {"diagnostic_only", "exploratory_e2e_authorized"})


def test_retry19_trajectory_inventory_is_exactly_fifteen_and_old_is_regression() -> None:
    config = load_config(CONFIG)
    inventory = config["trajectories"]["inventory"]
    assert sum(inventory.values()) == 15
    assert config["trajectories"]["waypoint_count"] == 120
    assert inventory == {
        "old_rounded_rectangles": 3,
        "seam_between_rectangles": 3,
        "exact_seam_waypoint_rectangles": 3,
        "slanted_ellipse": 1,
        "star": 1,
        "spiral_3d": 1,
        "zero_to_boundary": 3,
    }


def test_trajectory_teacher_restores_waypoint_order_on_candidate_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(CONFIG)
    trajectory_id = "trajectory"
    target_id = f"{trajectory_id}:00000"
    stage = tmp_path / STAGE_DIRS["postlock_trajectories"]
    stage.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "trajectory_id": trajectory_id,
                "waypoint_index": 0,
                "x_m": 1.2,
                "y_m": 0.0,
                "z_m": 0.0,
            }
        ]
    ).to_parquet(stage / "postlock_15_trajectory_waypoints.parquet", index=False)
    candidate = pd.DataFrame(
        [
            {
                "target_id": target_id,
                "candidate_id": "candidate",
                **{column: 0.0 for column in retry19.BETA_COLUMNS},
            }
        ]
    )

    monkeypatch.setattr(retry19, "_gate", lambda *_args: {"status": "complete"})
    monkeypatch.setattr(retry19, "_verify_lock", lambda *_args: True)
    monkeypatch.setattr(retry19.retry17, "_solve_candidates", lambda *_args, **_kwargs: candidate.copy())

    def cycle_teacher(candidates, **_kwargs):
        assert candidates["waypoint_index"].tolist() == [0]
        assert candidates["trajectory_id"].tolist() == [trajectory_id]
        return candidates.copy()

    monkeypatch.setattr(retry19.retry17, "_cycle_teacher", cycle_teacher)

    gate = retry19.stage_trajectory_teacher(config, tmp_path, smoke=False)

    assert gate["status"] == "complete"
    teacher = pd.read_parquet(
        tmp_path / STAGE_DIRS["trajectory_teacher"] / "trajectory_teacher_labels.parquet"
    )
    assert teacher["target_id"].tolist() == [target_id]


def test_trajectory_teacher_gap_is_diagnostic_for_raw_and_dls_evaluation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(CONFIG)
    stage = tmp_path / STAGE_DIRS["postlock_trajectories"]
    stage.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "trajectory_id": "new_rectangle",
                "waypoint_index": 0,
                "shape_class": "new_rectangle",
                "trajectory_role": "seam_between_rectangles",
                "x_m": 1.2,
                "y_m": 0.0,
                "z_m": 0.0,
            }
        ]
    ).to_parquet(stage / "postlock_15_trajectory_waypoints.parquet", index=False)
    lock_stage = tmp_path / STAGE_DIRS["student_lock"]
    lock_stage.mkdir(parents=True)
    (lock_stage / "dataset_teacher_split_model_lock.json").write_text("{}\n", encoding="utf-8")

    class Environment:
        @staticmethod
        def fk(_beta):
            return np.zeros(3)

    class Model:
        @staticmethod
        def __call__(xyz, *, training):
            assert training is False
            return np.zeros((len(xyz), 6))

    metric = {
        "fk_p95_mm": 0.0,
        "fk_maximum_mm": 0.0,
        "path_step_excess_maximum_mm": 0.0,
        "raw_step_gt7_rate": 0.0,
    }
    monkeypatch.setattr(retry19, "_gate", lambda *_args: {"status": "teacher_red"})
    monkeypatch.setattr(retry19, "_verify_lock", lambda *_args: True)
    monkeypatch.setattr(retry19, "_environment", lambda *_args: Environment())
    monkeypatch.setattr(
        retry19,
        "_load_selected_models",
        lambda *_args: {int(config["student"]["primary_seed"]): Model()},
    )
    monkeypatch.setattr(
        retry19.retry17,
        "_two_step_dls",
        lambda _env, raw, _xyz, *, zero_xyz: raw,
    )
    monkeypatch.setattr(retry19.retry18, "_path_metrics", lambda *_args, **_kwargs: metric.copy())

    gate = retry19.stage_trajectory_evaluation(config, tmp_path, smoke=False)

    assert gate["status"] == "complete"
    assert gate["trajectory_teacher_complete"] is False
    assert gate["trajectory_teacher_completeness_is_diagnostic"] is True
    report = pd.read_parquet(tmp_path / STAGE_DIRS["trajectory_evaluation"] / "trajectory_report.parquet")
    assert len(report) == 1
    assert report["dls2_success_rate"].item() == 0.0


def test_retry19_audit_only_rows_are_never_student_splits() -> None:
    from quasi_exp.teacher.retry19_direct_student import unified_split_registry

    rows = pd.DataFrame(
        [
            {"target_id": "primary", "x_m": 1.0, "y_m": 0.0, "z_m": 0.0, "domain_class": "primary"},
            {"target_id": "audit", "x_m": 1.0, "y_m": 0.0, "z_m": 0.0, "domain_class": "audit_only"},
        ]
    )
    assigned, _registry = unified_split_registry(rows, pd.DataFrame())
    assert assigned.set_index("target_id").loc["audit", "split_role"] == "audit_only"


def test_retry19_writes_complete_objective_feasibility_contract(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    cells = pd.DataFrame(
        [
            {
                "cell_id": "a",
                "volume_mm3": 125.0,
                "probe_x_m": 1.215,
                "probe_y_m": 0.0,
                "probe_z_m": 0.0,
            },
            {
                "cell_id": "b",
                "volume_mm3": 125.0,
                "probe_x_m": 1.220,
                "probe_y_m": 0.0,
                "probe_z_m": 0.0,
            },
        ]
    )
    targets = pd.DataFrame(
        [
            {
                "target_id": "target",
                "target_role": "primary_cell_service",
                "x_m": 1.2175,
                "y_m": 0.0,
                "z_m": 0.0,
            }
        ]
    )
    verification = pd.DataFrame(
        [
            {
                "source": "upstream",
                "path": "/tmp/upstream",
                "expected_sha256": "0" * 64,
                "observed_sha256": "0" * 64,
                "verified": True,
            }
        ]
    )

    gate = _write_objective_feasibility_contract(
        config,
        tmp_path,
        required_primary=cells,
        primary_targets=targets,
        verification=verification,
        proposal_only_count=2,
    )

    contract = tmp_path / "00_objective_feasibility"
    required = {
        "objective_contract.json",
        "denominator_size.json",
        "reusable_evidence.json",
        "budget_lower_bound.json",
        "atomic_objective_schedule.json",
        "gate.json",
    }
    assert required <= {path.name for path in contract.iterdir()}
    denominator = json.loads((contract / "denominator_size.json").read_text())
    lower = json.loads((contract / "budget_lower_bound.json").read_text())
    reusable = json.loads((contract / "reusable_evidence.json").read_text())
    assert denominator["frozen_before_launch"] is True
    assert denominator["denominators"][0]["required_count"] == 2
    assert lower["objective_lower_bounds"][0]["target_units"] == 2
    assert lower["objective_lower_bounds"][0]["minimum_resources"]["new_supervision_vertices"] == 1
    assert reusable["ineligible_counts"]["proposal_only"] == 2
    assert gate["claim_bearing_run_authorized"] is True


def test_retry19_identity_uses_registry_source_not_binding_commit(tmp_path: Path, monkeypatch) -> None:
    config = load_config(CONFIG)
    source_sha = "a" * 40
    binding_sha = "b" * 40
    definition = {
        "scientific_source_fixed_point": source_sha,
        "protocol_sources": [config["sources"]["governing_protocol"]],
        "config": str(CONFIG.relative_to(ROOT)),
        "runner": "scripts/analysis/run_bacra_v14_3r_retry19_fullspace_direct_student.py",
        "launcher": "scripts/pipelines/run_bacra_v14_3r_retry19_fullspace_direct_student.sh",
        "tests": {"self_contained": ["tests/test_bacra_v14_3r_retry19_runner.py"]},
    }
    monkeypatch.setattr(retry19, "_binding_definition", lambda _sha: definition)
    monkeypatch.setattr(retry19, "_checkout_sha", lambda: binding_sha)
    monkeypatch.setattr(
        retry19.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    monkeypatch.setattr(retry19, "_ACTIVE_SCIENTIFIC_SOURCE_SHA", None)

    retry19._ensure_identity(config, tmp_path / "attempt", binding_sha, smoke=False)

    identity = json.loads((tmp_path / "attempt" / "run_identity.json").read_text())
    assert identity["scientific_source_fixed_point"] == source_sha
    assert identity["binding_fixed_point"] == binding_sha
