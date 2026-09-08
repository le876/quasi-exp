import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/analysis/run_bacra_retry20_preflight.py"
spec = importlib.util.spec_from_file_location("retry20_preflight", PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_manifest_paths_are_relative_to_stage_and_tampering_rejected(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    data = stage / "data.json"
    data.write_text("{}")
    manifest = stage / "completion_manifest.json"
    manifest.write_text(json.dumps({"artifacts": [{"path": "data.json", "sha256": runner.digest(data)}]}))
    config = {"upstream_root": str(tmp_path), "upstream_manifests": {"stage/completion_manifest.json": runner.digest(manifest)}}
    _, records = runner.verify_upstream(config)
    assert records[0]["path"] == str(data)
    data.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed upstream artifact"):
        runner.verify_upstream(config)


def test_invalid_identity_never_creates_output(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "identity", lambda *args: (_ for _ in ()).throw(ValueError("invalid identity")))
    with pytest.raises(ValueError, match="invalid identity"):
        runner.run({}, tmp_path/"config", tmp_path/"output", "bad")
    assert not (tmp_path/"output").exists()


def test_strict_json_serialization_does_not_emit_nan(tmp_path):
    p = tmp_path/"audit.json"
    runner.write_json(p, {"undefined_correlation": float("nan")})
    assert json.loads(p.read_text()) == {"undefined_correlation": None}


def test_attempt1_budget_feasibility_does_not_override_failed_geometry():
    # Artifact-bound minimal reproduction: attempt1 had 26389 admissible coarse
    # replacements and failed coverage. Its shared budget status was misencoded.
    gate = runner.budget_gate({"experiment_id": runner.EXPERIMENT}, 26389, 24000, False)
    assert gate["status"] == "feasible"
    assert gate["claim_bearing_run_authorized"]
    assert not gate["geometry_gate_passed"]
    assert not gate["full_experiment_authorized"]
    assert not gate["reason_codes"]
    gate = runner.budget_gate({}, 23999, 24000, False)
    assert gate["status"] == "diagnostic_only"
    assert not gate["claim_bearing_run_authorized"]
    assert gate["failed_objective_ids"] == ["coarse_relocation"]
