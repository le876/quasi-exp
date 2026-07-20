#!/usr/bin/env python3
"""Fail closed if any deployment artifact is missing or hash-mismatched."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quasi_exp.teacher.experiment import atomic_write_json, sha256_file


def verify(path: str | Path, expected_sha256: str) -> dict[str, str]:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(str(artifact))
    actual = sha256_file(artifact)
    if actual != str(expected_sha256):
        raise RuntimeError(f"artifact hash mismatch: {artifact}")
    return {"path": str(artifact.resolve()), "sha256": actual}


def run(summary_path: Path) -> dict[str, object]:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    root = summary_path.resolve().parents[1]
    checked = [
        verify(root / "screen/selection.json", summary["selection_sha256"]),
        verify(root / "final/final_summary.json", summary["final_summary_sha256"]),
        verify(root / "gpu_preflight.json", summary["gpu_preflight_sha256"]),
    ]
    for path, digest in summary["worker_code_sha256"].items():
        checked.append(verify(path, digest))
    for report in summary["reports"]:
        checked.append(verify(report["model_path"], report["model_sha256"]))
        for scale in report["scales"].values():
            checked.append(verify(scale["prediction_path"], scale["prediction_sha256"]))
            plot = Path(scale["plot_path"])
            if not plot.is_file() or not plot.with_suffix(".pdf").is_file():
                raise FileNotFoundError(f"missing PNG/PDF tracking plot pair: {plot}")
    result: dict[str, object] = {
        "protocol_id": "large-scale-student-tracking-v10.1-artifact-verification",
        "verification_pass": True,
        "hashed_artifact_count": len(checked),
        "checked": checked,
    }
    atomic_write_json(root / "artifact_verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.summary.resolve()), indent=2))
