from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTER = ROOT / "scripts/analysis/report_bacra_v14_2r_patch_benchmark.py"


def _module():
    spec = importlib.util.spec_from_file_location("bacra_v14_2r_benchmark", REPORTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_patch_benchmark_separates_performance_from_scientific_gate(
    tmp_path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_git_sha", lambda: "source-a")
    monkeypatch.setattr(module, "_tree_clean", lambda: True)
    config = tmp_path / "config.yaml"
    config.write_text(
        "audit_execution:\n  total_shard_workers: 12\n", encoding="utf-8"
    )
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    _write_json(
        tmp_path / "00_inventory/source_fixed_point.json",
        {"source_sha": "source-a", "config_sha256": config_sha},
    )
    _write_json(
        tmp_path / "03_rooted_baseline/patch_07/baseline/report.json",
        {
            "patch_id": "patch_07",
            "variant": "baseline",
            "runtime_s": 6400.0,
            "gate_pass": False,
        },
    )
    for phase in ("chart_initial", "primary_certificate"):
        _write_json(
            tmp_path
            / f"03_rooted_baseline/patch_07/baseline/_audit_checkpoints/{phase}/gate.json",
            {"gate_pass": True, "shard_count": 12},
        )

    report = module.build_report(
        output_root=tmp_path,
        config_path=config,
        runtime_per_parent_max_s=120.0,
    )

    assert report["gate_pass"] is True
    assert report["runtime_per_parent_s"] == 100.0
    assert report["scientific_patch_gate"] is False
    assert report["scientific_patch_gate_is_not_a_performance_prerequisite"] is True


def test_patch_benchmark_fails_when_a_required_phase_is_absent(
    tmp_path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_git_sha", lambda: "source-a")
    monkeypatch.setattr(module, "_tree_clean", lambda: True)
    config = tmp_path / "config.yaml"
    config.write_text(
        "audit_execution:\n  total_shard_workers: 12\n", encoding="utf-8"
    )
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    _write_json(
        tmp_path / "00_inventory/source_fixed_point.json",
        {"source_sha": "source-a", "config_sha256": config_sha},
    )
    _write_json(
        tmp_path / "03_rooted_baseline/patch_07/baseline/report.json",
        {
            "patch_id": "patch_07",
            "variant": "baseline",
            "runtime_s": 100.0,
            "gate_pass": True,
        },
    )
    _write_json(
        tmp_path
        / "03_rooted_baseline/patch_07/baseline/_audit_checkpoints/chart_initial/gate.json",
        {"gate_pass": True, "shard_count": 12},
    )

    report = module.build_report(
        output_root=tmp_path,
        config_path=config,
        runtime_per_parent_max_s=120.0,
    )

    assert report["gate_pass"] is False
    assert report["checks"]["required_audit_phases"] is False
