from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bacra_v14_3r_retry15_omega600_domain_discovery.yaml"
PROTOCOL = ROOT / "docs/protocols/32-BACRA-V14.3R-retry15-Omega600-canonical-graph执行协议.md"
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_3r_retry15_discovery.py"
WORKER = ROOT / "scripts/analysis/run_bacra_v14_3r_retry15_worker.py"


def _module():
    spec = importlib.util.spec_from_file_location("retry15_discovery_runner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discovery_config_freezes_omega600_and_independent_seeds() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["experiment_id"] == "bacra_v14_3r_retry15_omega600_domain_discovery"
    assert config["omega600"] == {
        "zero_x_m": 1.2154980000000004,
        "minimum_x_m": 0.6154980000000004,
        "axial_length_mm": 600.0,
    }
    assert config["proposal"]["pool_a_seed"] == 20260895
    assert config["proposal"]["pool_b_seed"] == 20260896
    assert config["teacher_seed_banks"]["full"]["seed"] == 20260897
    assert config["teacher_seed_banks"]["y_seam"]["seed"] == 20260898
    assert config["teacher_seed_banks"]["z_seam"]["seed"] == 20260899
    assert len(
        {
            config["proposal"]["pool_a_seed"],
            config["proposal"]["pool_b_seed"],
            *(value["seed"] for value in config["teacher_seed_banks"].values()),
        }
    ) == 5
    assert not config["claims"]["claim_bearing_run_authorized"]
    assert config["claims"]["diagnostic_only"]


def test_protocol_keeps_outer_denominator_and_large_circle_atomic() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "0.6154980000000004" in text
    assert "经验外部分母" in text
    assert "required voxel 不得" in text
    assert "三个外圈都必须不小于 `100 mm`" in text
    assert "proposal beta 只用于 FK" in text
    assert "full workspace" in text


def test_stage_order_freezes_domain_before_teacher_preflight() -> None:
    module = _module()
    assert module.STAGE_ORDER == (
        "inventory",
        "workspace_probe",
        "domain_selection",
        "seed_banks",
        "teacher_preflight",
        "summary",
    )


def test_runner_rejects_old_interval_or_claim_authorization(tmp_path: Path) -> None:
    module = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["omega600"]["axial_length_mm"] = 200.0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_config(path)

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["claims"]["claim_bearing_run_authorized"] = True
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic-only"):
        module.load_config(path)


def test_worker_receives_dedicated_full_and_seam_seed_banks() -> None:
    text = WORKER.read_text(encoding="utf-8")
    assert "--seed-bank" in text
    assert "--y-seed-bank" in text
    assert "--z-seed-bank" in text
    assert "proposal" not in " ".join(
        line for line in text.splitlines() if "add_argument" in line
    )


def test_launcher_requires_explicit_binding_sha() -> None:
    text = (
        ROOT / "scripts/pipelines/run_bacra_v14_3r_retry15_discovery.sh"
    ).read_text(encoding="utf-8")
    assert "BACRA_RETRY15_DISCOVERY_BINDING_SHA" in text
    assert "--binding-sha" in text
    assert "CUDA_VISIBLE_DEVICES=-1" in text
