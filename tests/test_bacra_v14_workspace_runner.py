from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_omega200_atlas.py"


def _module():
    spec = importlib.util.spec_from_file_location("bacra_v14_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _script_module(filename: str):
    path = ROOT / "scripts/analysis" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_presets_replace_reach_rounds_and_keep_fixed_budget_contract() -> None:
    module = _module()
    config_path = ROOT / "configs/bacra_v14_omega200_workspace_atlas.yaml"
    smoke = module.load_protocol_config(config_path, "smoke")
    pilot = module.load_protocol_config(config_path, "pilot")

    assert len(smoke["reach"]["rounds"]) == 2
    assert smoke["reach"]["rounds"][0]["power"] == 14
    assert len(pilot["reach"]["rounds"]) == 3
    assert pilot["budget"]["total_rows"] == 200_000
    assert pilot["budget"]["active_rows"] == 15_000
    assert pilot["registry"]["pilot_cell_count"] == 5_000
    assert pilot["candidates"]["saturation_fraction"] == pytest.approx(0.10)


def test_formal_preflight_is_fail_closed_without_successful_pilot_gate(tmp_path: Path) -> None:
    module = _module()
    config = module.load_protocol_config(
        ROOT / "configs/bacra_v14_omega200_workspace_atlas.yaml", "formal"
    )
    config["formal_requirements"]["pilot_summary"] = str(
        tmp_path / "missing_pilot_gate.json"
    )

    result = module.check_formal_prerequisites(config, project_root=tmp_path)

    assert result["gate_pass"] is False
    assert result["pilot_gate_exists"] is False
    assert "missing_pilot_gate" in result["reasons"]


def test_runner_declares_corrected_stages_before_student() -> None:
    module = _module()

    assert module.STAGE_ORDER == (
        "protocol",
        "workspace_proxy",
        "domain_registry",
        "candidate_bank",
        "workspace_atlas",
        "budget_allocation",
        "dataset",
        "student",
        "spatial_evaluation",
        "trajectory_evaluation",
        "summary",
    )


def test_implementation_source_inventory_binds_v14_runtime_bytes() -> None:
    module = _module()

    inventory = module.implementation_source_inventory(ROOT)

    assert "scripts/analysis/run_bacra_v14_omega200_atlas.py" in inventory
    assert "src/quasi_exp/teacher/workspace_candidate_bank.py" in inventory
    assert "src/quasi_exp/teacher/workspace_protocol.py" in inventory
    assert all(
        len(payload["sha256"]) == 64 and payload["size_bytes"] > 0
        for payload in inventory.values()
    )
    expected_root = module.implementation_source_root_sha256(inventory)
    assert len(expected_root) == 64
    assert expected_root == module.implementation_source_root_sha256(
        dict(reversed(tuple(inventory.items())))
    )


def test_protocol_persists_implementation_source_closure(tmp_path: Path) -> None:
    module = _module()
    config = module.load_protocol_config(
        ROOT / "configs/bacra_v14_omega200_workspace_atlas.yaml", "smoke"
    )

    gate = module.stage_protocol(
        config, module.project_root_from(ROOT), tmp_path
    )
    manifest = json.loads(
        (tmp_path / module.STAGE_DIRS["protocol"] / "source_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert gate["gate_pass"] is True
    assert gate["checks"]["implementation_source_inventory_nonempty"] is True
    assert manifest["source_manifest_schema_version"] == 2
    assert manifest["implementation_source_count"] == len(
        manifest["implementation_sources"]
    )
    assert manifest["implementation_source_root_sha256"] == gate[
        "implementation_source_root_sha256"
    ]
    assert manifest["implementation_sources"][
        "src/quasi_exp/teacher/workspace_candidate_bank.py"
    ]["sha256"] == module.sha256_file(
        ROOT / "src/quasi_exp/teacher/workspace_candidate_bank.py"
    )


def test_every_declared_stage_is_connected_to_a_real_runner() -> None:
    module = _module()

    assert set(module.STAGE_RUNNERS) == set(module.STAGE_ORDER)
    assert module.STAGE_RUNNERS["trajectory_evaluation"] is module.stage_trajectory_evaluation
    assert module.STAGE_RUNNERS["summary"] is module.stage_summary


def test_narrow_train_and_evaluation_entrypoints_expose_create_once_stage_selection() -> None:
    train = _script_module("train_bacra_v14_student.py")
    evaluate = _script_module("evaluate_bacra_v14_workspace.py")

    assert train.build_parser().parse_args([]).preset == "smoke"
    parsed = evaluate.build_parser().parse_args(["--stage", "trajectory_evaluation"])
    assert parsed.stage == "trajectory_evaluation"
    assert tuple(evaluate.EVALUATION_STAGES) == (
        "spatial_evaluation",
        "trajectory_evaluation",
        "summary",
    )


def test_registry_probe_schema_closes_atlas_cell_identity_contract() -> None:
    module = _module()
    cell = module.CellKey(10, 101, -17, -33)
    pilot = SimpleNamespace(
        cell=cell,
        representative_sample_id="representative",
        measure_sample_ids=("measure",),
    )
    samples = {
        "representative": SimpleNamespace(
            xyz_m=np.asarray([1.015, -0.165, -0.325]),
            beta_rad=np.arange(6, dtype=float),
        ),
        "measure": SimpleNamespace(
            xyz_m=np.asarray([1.019, -0.161, -0.321]),
            beta_rad=np.arange(6, dtype=float) + 0.1,
        ),
    }

    probes = module.pilot_probe_frame(
        (pilot,), sample_by_id=samples, node_id_by_cell={cell: 7}
    )

    assert probes[["cell_level_mm", "cell_ix", "cell_iy", "cell_iz"]].to_dict(
        "records"
    ) == [
        {"cell_level_mm": 10, "cell_ix": 101, "cell_iy": -17, "cell_iz": -33},
        {"cell_level_mm": 10, "cell_ix": 101, "cell_iy": -17, "cell_iz": -33},
    ]
    assert probes["is_representative"].tolist() == [True, False]


def test_chart_training_support_fails_closed_on_validation_only_expert() -> None:
    module = _module()
    frame = pd.DataFrame(
        {
            "chart_id": ["chart_a", "chart_a", "chart_b"],
            "split_role": ["train_core", "validation", "validation"],
        }
    )

    support = module.chart_training_support(frame)

    assert support["gate_pass"] is False
    assert support["unsupported_chart_ids"] == ["chart_b"]
    assert support["train_supported_chart_count"] == 1


@pytest.mark.parametrize(
    ("preset", "expected_scientific_gate", "expected_pilot_gate"),
    (("smoke", False, False), ("pilot", True, True)),
)
def test_summary_separates_operational_completion_from_pilot_scientific_gate(
    tmp_path: Path,
    preset: str,
    expected_scientific_gate: bool,
    expected_pilot_gate: bool,
) -> None:
    module = _module()
    config = module.load_protocol_config(
        ROOT / "configs/bacra_v14_omega200_workspace_atlas.yaml", preset
    )
    scientific_fields = {
        "workspace_proxy": {"reach_convergence_gate": True},
        "candidate_bank": {"branch_saturation_gate": True},
        "workspace_atlas": {
            "scientific_gate_pass": True,
            "branch_saturation_gate": True,
            "representation_gate_pass": True,
            "atlas_audit_gate_pass": True,
            "representation_mode": "xyz_global",
        },
        "budget_allocation": {
            "scientific_gate_pass": True,
            "budget_feasibility_gate": True,
        },
        "dataset": {"scientific_gate_pass": True, "target_reached": True},
        "student": {
            "scientific_gate_pass": True,
            "validation_gate": True,
        },
        "spatial_evaluation": {
            "scientific_gate_pass": True,
            "sealed_gate": True,
        },
        "trajectory_evaluation": {
            "scientific_gate_pass": True,
            "trajectory_gate": True,
        },
    }
    for name in module.STAGE_ORDER[:-1]:
        stage = tmp_path / module.STAGE_DIRS[name]
        stage.mkdir(parents=True)
        payload = {"gate_pass": True, **scientific_fields.get(name, {})}
        (stage / "gate.json").write_text(json.dumps(payload), encoding="utf-8")
        (stage / "evidence.txt").write_text(name, encoding="utf-8")
    protocol = tmp_path / module.STAGE_DIRS["protocol"]
    (protocol / "source_manifest.json").write_text(
        json.dumps({"source_git_sha": "fixed-point", "sources": {}}),
        encoding="utf-8",
    )

    result = module.stage_summary(config, ROOT, tmp_path)

    assert result["gate_pass"] is True
    assert result["scientific_gate_pass"] is expected_scientific_gate
    pilot = json.loads(
        (tmp_path / module.STAGE_DIRS["summary"] / "pilot_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert pilot["gate_pass"] is expected_pilot_gate
    assert pilot["reach_convergence_gate"] is True
    assert pilot["branch_saturation_gate"] is True
    assert pilot["representation_gate_pass"] is True


def test_saturation_audit_selection_is_exact_deterministic_and_stratified() -> None:
    module = _module()
    tasks = pd.DataFrame(
        {
            "node_id": np.arange(20),
            "x_bin": np.repeat(np.arange(4), 5),
            "strata": [
                "boundary|tip" if index % 2 == 0 else "interior|retention"
                for index in range(20)
            ],
        }
    )

    first = module.select_saturation_node_ids(tasks, fraction=0.10, seed=17)
    second = module.select_saturation_node_ids(tasks, fraction=0.10, seed=17)

    assert first == second
    assert len(first) == 2
    assert set(tasks.set_index("node_id").loc[list(first), "strata"]) == {
        "boundary|tip",
        "interior|retention",
    }


def test_diverse_seed_rows_are_bounded_unique_and_keep_node_identity() -> None:
    module = _module()
    capability_xyz = np.column_stack(
        [np.linspace(0.0, 0.01, 16), np.zeros(16), np.zeros(16)]
    )
    capability_beta = np.zeros((16, 6), dtype=float)
    capability_beta[:, 0] = np.linspace(-1.0, 1.0, 16)
    targets = pd.DataFrame(
        {"node_id": [7], "x_m": [0.005], "y_m": [0.0], "z_m": [0.0]}
    )
    bounds = np.tile(np.asarray([[-1.0, 1.0]]), (6, 1))

    seeds = module.diverse_capability_seed_rows(
        targets,
        capability_xyz=capability_xyz,
        capability_beta=capability_beta,
        bounds_rad=bounds,
        seed_count=4,
        neighbor_count=16,
    )

    assert len(seeds) == 4
    assert set(seeds["node_id"]) == {7}
    assert seeds["seed_rank"].tolist() == [0, 1, 2, 3]
    assert seeds.loc[:, [f"beta{i}_rad" for i in range(1, 7)]].drop_duplicates().shape[0] == 4
