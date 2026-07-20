from __future__ import annotations

import json
from pathlib import Path

from quasi_exp.teacher.region_artifacts import (
    finalize_experiment,
    verify_completed_experiment,
)


def test_final_verifier_is_last_writer_and_detects_later_artifact_mutation(tmp_path: Path) -> None:
    stage = tmp_path / "00_protocol"
    stage.mkdir()
    (stage / "gate.json").write_text(
        json.dumps({"gate_pass": True, "checks": {"frozen": True}}),
        encoding="utf-8",
    )
    (stage / "data.bin").write_bytes(b"evidence")

    verification = finalize_experiment(
        tmp_path,
        required_gate_files=(Path("00_protocol/gate.json"),),
    )

    assert verification["artifact_verification_pass"] is True
    assert (tmp_path / "artifact_verification.json").exists()
    assert (tmp_path / "COMPLETED").exists()
    assert (tmp_path / "COMPLETED").stat().st_mtime_ns >= (
        tmp_path / "artifact_verification.json"
    ).stat().st_mtime_ns
    assert verify_completed_experiment(tmp_path)["artifact_verification_pass"] is True

    (stage / "data.bin").write_bytes(b"changed-after-completion")
    report = verify_completed_experiment(tmp_path)
    assert report["artifact_verification_pass"] is False
    assert "00_protocol/data.bin" in report["changed_files"]
