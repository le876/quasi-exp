from __future__ import annotations

from pathlib import Path

import numpy as np

from quasi_exp.teacher.experiment import (
    ExperimentManifest,
    atomic_write_json,
    sha256_file,
)


def test_manifest_fingerprint_binds_protocol_inputs_and_raw_worker_bytes(tmp_path: Path) -> None:
    worker = tmp_path / "worker.parquet"
    worker.write_bytes(b"raw-worker-bytes")
    manifest = ExperimentManifest.create(
        protocol_id="trajectory-canonical-teacher-v10.1",
        teacher_variants=("T0", "T1", "T3"),
        family_id="F1",
        radii_mm=(92.5, 100.0),
        phase_count=180,
        tube_offsets_mm=(-1.0, 0.0, 1.0),
        solver_seed=20260720,
        input_files=(worker,),
        worker_code_files=(Path(__file__),),
    )

    assert manifest.input_sha256[str(worker.resolve())] == sha256_file(worker)
    assert len(manifest.protocol_sha256) == 64
    assert manifest.raw_worker_bytes_sha256 == sha256_file(worker)
    assert manifest.worker_code_sha256[str(Path(__file__).resolve())] == sha256_file(__file__)


def test_atomic_json_rejects_non_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    atomic_write_json(path, {"ok": 1.0})
    assert path.read_text(encoding="utf-8").endswith("\n")

    try:
        atomic_write_json(path, {"bad": np.nan})
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite JSON must be rejected")
