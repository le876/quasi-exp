from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/analysis/run_generalized_ellipse_region_v11.py"
    spec = importlib.util.spec_from_file_location("region_runner_v11", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_stage_freezes_catalog_splits_and_manifest(tmp_path: Path) -> None:
    runner = _runner_module()
    source_root = Path(__file__).resolve().parents[1]
    project_root = runner.project_root_from(source_root)
    config = runner.load_protocol_config(
        source_root / "configs/generalized_ellipse_region_v11.yaml",
        preset="smoke",
    )

    report = runner.run_protocol_stage(
        config=config,
        source_root=source_root,
        project_root=project_root,
        output=tmp_path,
    )

    assert report["gate_pass"] is True
    assert all(type(value) is bool for value in report["checks"].values())
    catalog = pd.read_csv(tmp_path / "00_protocol/family_catalog.csv")
    splits = pd.read_csv(tmp_path / "00_protocol/split_manifest.csv")
    manifest = json.loads((tmp_path / "00_protocol/artifact_manifest.json").read_text())
    assert len(catalog) == 72
    assert len(splits) == 24
    assert catalog["family_id"].is_unique
    assert len(manifest["protocol_sha256"]) == 64
    assert manifest["source_fixed_point"] == config["source_fixed_point"]


def test_stage_cache_recomputes_boolean_gate_instead_of_trusting_cached_flag(tmp_path: Path) -> None:
    runner = _runner_module()
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "gate_pass": True,
                "checks": {"residual": False, "margin": True},
            }
        ),
        encoding="utf-8",
    )

    assert runner.read_valid_gate(gate) is None


def test_stage_cache_is_invalidated_when_a_hashed_artifact_changes(tmp_path: Path) -> None:
    runner = _runner_module()
    artifact = tmp_path / "surface.parquet"
    artifact.write_bytes(b"frozen")
    gate_path = tmp_path / "gate.json"
    runner._write_gate(gate_path, checks={"surface_complete": True})

    assert runner.read_valid_gate(gate_path) is not None
    artifact.write_bytes(b"mutated")
    assert runner.read_valid_gate(gate_path) is None
