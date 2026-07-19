"""Frozen experiment manifests and crash-safe evidence writes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ExperimentManifest:
    protocol_id: str
    teacher_variants: tuple[str, ...]
    family_id: str
    radii_mm: tuple[float, ...]
    phase_count: int
    tube_offsets_mm: tuple[float, ...]
    solver_seed: int
    traversal_direction: str
    cyclic_cut: int
    input_sha256: dict[str, str]
    raw_worker_bytes_sha256: str
    worker_code_sha256: dict[str, str]
    protocol_sha256: str

    @classmethod
    def create(
        cls,
        *,
        protocol_id: str,
        teacher_variants: Sequence[str],
        family_id: str,
        radii_mm: Sequence[float],
        phase_count: int,
        tube_offsets_mm: Sequence[float],
        solver_seed: int,
        traversal_direction: str = "forward",
        cyclic_cut: int = 0,
        input_files: Sequence[str | Path],
        worker_code_files: Sequence[str | Path] = (),
    ) -> "ExperimentManifest":
        files = tuple(sorted(Path(path).resolve() for path in input_files))
        input_hashes = {str(path): sha256_file(path) for path in files}
        worker_files = tuple(sorted(Path(path).resolve() for path in worker_code_files))
        worker_hashes = {str(path): sha256_file(path) for path in worker_files}
        raw_digest = hashlib.sha256()
        for path in files:
            raw_digest.update(path.read_bytes())
        payload = {
            "protocol_id": str(protocol_id),
            "teacher_variants": tuple(str(value) for value in teacher_variants),
            "family_id": str(family_id),
            "radii_mm": tuple(float(value) for value in radii_mm),
            "phase_count": int(phase_count),
            "tube_offsets_mm": tuple(float(value) for value in tube_offsets_mm),
            "solver_seed": int(solver_seed),
            "traversal_direction": str(traversal_direction),
            "cyclic_cut": int(cyclic_cut),
            "input_sha256": input_hashes,
            "raw_worker_bytes_sha256": raw_digest.hexdigest(),
            "worker_code_sha256": worker_hashes,
        }
        return cls(**payload, protocol_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(payload) + b"\n"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
