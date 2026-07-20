"""Final-write artifact verification for resumable V11 experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .experiment import atomic_write_json
from .region_protocol import require_boolean_gate_tree


_CONTROL_FILES = {"artifact_verification.json", "COMPLETED"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.relative_to(root).as_posix() in _CONTROL_FILES:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return rows


def finalize_experiment(
    root: str | Path,
    *,
    required_gate_files: Sequence[Path],
) -> dict[str, Any]:
    output = Path(root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPLETED").exists():
        raise RuntimeError("experiment is already completed and immutable")
    gates = []
    for relative in required_gate_files:
        path = (output / relative).resolve()
        if output not in path.parents or not path.is_file():
            raise FileNotFoundError(str(relative))
        payload = json.loads(path.read_text(encoding="utf-8"))
        gate_pass = require_boolean_gate_tree(payload)
        gates.append({"path": Path(relative).as_posix(), "gate_pass": gate_pass})
    if not all(row["gate_pass"] for row in gates):
        raise RuntimeError("cannot finalize an experiment with a failed hard gate")
    artifacts = _artifact_rows(output)
    report = {
        "schema_version": 1,
        "required_gates": gates,
        "artifacts": artifacts,
        "artifact_count": int(len(artifacts)),
        "artifact_verification_pass": True,
        "changed_files": [],
        "missing_files": [],
        "unexpected_files": [],
    }
    atomic_write_json(output / "artifact_verification.json", report)
    verification_sha = _sha256(output / "artifact_verification.json")
    (output / "COMPLETED").write_text(
        json.dumps(
            {"artifact_verification_sha256": verification_sha},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def verify_completed_experiment(root: str | Path) -> dict[str, Any]:
    output = Path(root).resolve()
    verification_path = output / "artifact_verification.json"
    sentinel_path = output / "COMPLETED"
    if not verification_path.is_file() or not sentinel_path.is_file():
        return {
            "artifact_verification_pass": False,
            "reason": "missing_verification_or_completed_sentinel",
            "changed_files": [],
            "missing_files": [],
            "unexpected_files": [],
        }
    frozen = json.loads(verification_path.read_text(encoding="utf-8"))
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    control_valid = sentinel.get("artifact_verification_sha256") == _sha256(
        verification_path
    )
    expected = {row["path"]: row for row in frozen.get("artifacts", [])}
    actual = {row["path"]: row for row in _artifact_rows(output)}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    changed = sorted(
        path
        for path in set(expected) & set(actual)
        if expected[path]["sha256"] != actual[path]["sha256"]
        or int(expected[path]["bytes"]) != int(actual[path]["bytes"])
    )
    return {
        **frozen,
        "artifact_verification_pass": bool(
            control_valid and not missing and not unexpected and not changed
        ),
        "control_files_valid": bool(control_valid),
        "changed_files": changed,
        "missing_files": missing,
        "unexpected_files": unexpected,
    }
