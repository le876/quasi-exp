#!/usr/bin/env python3
"""Seal or verify the BACRA V14.2R patch-07 audit-shard benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True
    ).strip()


def _tree_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=SOURCE_ROOT, text=True
    ).strip()


def build_report(
    *, output_root: Path, config_path: Path, runtime_per_parent_max_s: float | None
) -> dict[str, Any]:
    patch_directory = output_root / "03_rooted_baseline/patch_07/baseline"
    patch_report = _read_json(patch_directory / "report.json")
    fixed = _read_json(output_root / "00_inventory/source_fixed_point.json")
    checkpoint_root = patch_directory / "_audit_checkpoints"
    phase_gates = {
        path.parent.name: _read_json(path)
        for path in sorted(checkpoint_root.glob("*/gate.json"))
    }
    required_phases = {"chart_initial", "primary_certificate"}
    parent_count = 64
    runtime_s = float(patch_report["runtime_s"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    total_shards = int(config["audit_execution"]["total_shard_workers"])
    runtime_limit = (
        float(runtime_per_parent_max_s)
        if runtime_per_parent_max_s is not None
        else float(
            config["audit_execution"][
                "patch_benchmark_runtime_per_parent_max_s"
            ]
        )
    )
    checks = {
        "clean_fixed_point": _tree_clean(),
        "source_sha": str(fixed.get("source_sha", "")) == _git_sha(),
        "config_sha256": str(fixed.get("config_sha256", ""))
        == _sha256(config_path),
        "patch_identity": str(patch_report.get("patch_id", "")) == "patch_07",
        "baseline_identity": str(patch_report.get("variant", "")) == "baseline",
        "required_audit_phases": required_phases <= set(phase_gates),
        "all_audit_phase_exact_sets": bool(phase_gates)
        and all(bool(gate.get("gate_pass", False)) for gate in phase_gates.values()),
        "twelve_shards_per_phase": bool(phase_gates)
        and all(int(gate.get("shard_count", 0)) == total_shards == 12 for gate in phase_gates.values()),
        "runtime_per_parent": runtime_s / parent_count
        <= runtime_limit,
    }
    return {
        "schema_version": 1,
        "gate_pass": bool(all(checks.values())),
        "checks": checks,
        "source_sha": _git_sha(),
        "config_sha256": _sha256(config_path),
        "patch_id": "patch_07",
        "variant": "baseline",
        "parent_count": parent_count,
        "runtime_s": runtime_s,
        "runtime_per_parent_s": runtime_s / parent_count,
        "runtime_per_parent_max_s": runtime_limit,
        "phase_gates": phase_gates,
        "scientific_patch_gate": bool(patch_report.get("gate_pass", False)),
        "scientific_patch_gate_is_not_a_performance_prerequisite": True,
        "patch_report_path": str(patch_directory / "report.json"),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--config",
        default=str(SOURCE_ROOT / "configs/bacra_v14_2r_stitched_atlas.yaml"),
    )
    parser.add_argument("--runtime-per-parent-max-s", type=float)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    report = build_report(
        output_root=output_root,
        config_path=Path(args.config).resolve(),
        runtime_per_parent_max_s=args.runtime_per_parent_max_s,
    )
    if not args.verify_only:
        _write_json(output_root / "benchmark/gate.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 1 if args.require_pass and not report["gate_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
