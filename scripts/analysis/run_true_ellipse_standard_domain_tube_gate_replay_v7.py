#!/usr/bin/env python3
"""Replay the V7 tube label gate over immutable, hash-bound tube evidence.

This runner intentionally does not rerun IK or surface optimization.  It
recomputes the artifact-local geometry and joint-margin metrics, requires the
original four-cut/cut-invariance evidence, applies the currently registered
tube success floor, and then materializes a fresh dataset/challenge chain in
an isolated output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parents[1] if REPO_ROOT.parent.name == ".worktrees" else REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import run_true_ellipse_standard_domain_v7 as v7  # noqa: E402
from true_ellipse_family_v5_utils import stable_fingerprint  # noqa: E402


DEFAULT_SOURCE_DIR = PROJECT_ROOT / "runs" / "true_ellipse_standard_domain_v7_gate_v2"
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs" / "true_ellipse_standard_domain_v7_gate_v2_99pct"
REPLAY_STRATEGY_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay V7's registered tube gate without rerunning tube optimization."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--v5-dir", type=Path, default=v7.DEFAULT_V5_DIR)
    parser.add_argument("--robot-config", type=Path, default=v7.DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def build_v7_args(args: argparse.Namespace) -> argparse.Namespace:
    source = Path(args.source_dir).resolve()
    output = Path(args.out_dir).resolve()
    if source == output:
        raise ValueError("tube gate replay output must be isolated from its source")
    argv = [
        "--preset",
        "formal",
        "--out-dir",
        str(output),
        "--v5-dir",
        str(Path(args.v5_dir).resolve()),
        "--robot-config",
        str(Path(args.robot_config).resolve()),
        "--workers",
        str(max(1, int(args.workers))),
    ]
    if bool(args.skip_existing):
        argv.append("--skip-existing")
    return v7.parse_args(argv)


def compute_replayed_tube_frontier(
    *,
    radial_frontier_mm: float,
    materialized_radii_mm: Iterable[float],
    passed_by_radius: Mapping[float, bool],
) -> dict[str, Any]:
    return v7.complete_tube_checkpoint_frontier(
        radial_frontier_mm=float(radial_frontier_mm),
        materialized_radii_mm=materialized_radii_mm,
        passed_by_radius=passed_by_radius,
    )


def _require_hash_bound_file(
    raw_path: str | Path | None,
    expected_sha256: str | None,
    *,
    label: str,
) -> Path:
    path = Path(str(raw_path or ""))
    expected = str(expected_sha256 or "")
    if not path.is_file() or not expected:
        raise ValueError(f"missing hash-bound {label}")
    current = v7.file_sha256(path)
    if current != expected:
        raise ValueError(f"hash mismatch for {label}")
    return path.resolve()


def _source_reports(source_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    paths = {
        "radial": source_dir / "01_radial" / "radial_report.json",
        "tube": source_dir / "02_tube" / "tube_report.json",
    }
    if not paths["radial"].is_file():
        raise FileNotFoundError(f"tube replay source radial report is missing: {paths['radial']}")
    radial = v7.read_json(paths["radial"])
    tube = v7.read_json(paths["tube"]) if paths["tube"].is_file() else {}
    if not bool(radial.get("downstream_radial_admission_gate_pass", False)):
        raise ValueError("source radial report is not admitted for downstream evidence")
    if radial.get("strict_geometry_rmax_mm") is None:
        raise ValueError("source radial report has no strict frontier")
    if tube and str(tube.get("radial_task_fingerprint", "")) != str(
        radial.get("task_fingerprint", "")
    ):
        raise ValueError("source tube report is not bound to the source radial report")
    frontier = float(radial["strict_geometry_rmax_mm"])
    for raw_radius, raw_entry in (radial.get("path_manifest") or {}).items():
        if float(raw_radius) > frontier + 1.0e-9:
            continue
        entry = dict(raw_entry)
        _require_hash_bound_file(
            entry.get("path"),
            entry.get("sha256"),
            label=f"radial centerline {raw_radius} mm",
        )
    return radial, tube, {
        key: v7.file_sha256(path) for key, path in paths.items() if path.is_file()
    }


def _quality_report_path(source_dir: Path, radius_mm: float) -> Path:
    return (
        source_dir
        / "02_tube"
        / v7.v6_runner.radius_slug(float(radius_mm))
        / "tube_quality_report.json"
    )


def _replay_radius(
    v7_args: argparse.Namespace,
    *,
    source_dir: Path,
    radius_mm: float,
    replay_report_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    report_path = _quality_report_path(source_dir, float(radius_mm))
    if not report_path.is_file():
        raise FileNotFoundError(f"source tube quality report is missing: {report_path}")
    source = v7.read_json(report_path)
    if not np.isclose(float(source.get("radius_mm", np.nan)), float(radius_mm)):
        raise ValueError(f"source tube radius mismatch at {radius_mm:g} mm")
    artifact = _require_hash_bound_file(
        source.get("tube_artifact_path"),
        source.get("tube_artifact_sha256"),
        label=f"tube artifact {radius_mm:g} mm",
    )
    centerline = _require_hash_bound_file(
        source.get("centerline_path"),
        source.get("centerline_sha256"),
        label=f"tube centerline {radius_mm:g} mm",
    )
    tube = pd.read_parquet(artifact)
    required = {
        "angle_idx",
        "tube_offset_id",
        "sample_id",
        "tube_success",
        "xyz_residual_mm",
        *v7.v6_utils.atlas.BETA_COLS,
    }
    missing = sorted(required - set(tube.columns))
    if missing:
        raise ValueError(f"tube replay artifact missing columns at {radius_mm:g} mm: {missing}")
    success = tube["xyz_residual_mm"].to_numpy(dtype=float) <= 1.5
    if not np.array_equal(success, tube["tube_success"].fillna(False).to_numpy(dtype=bool)):
        raise ValueError(f"tube_success is stale at {radius_mm:g} mm")
    offsets = tuple(v7.parse_float_csv(v7_args.tube_offsets_mm))
    geometric = v7.v6_runner._tube_quality_metrics(
        tube,
        final_points=int(v7_args.final_points),
        offsets_mm=offsets,
    )
    margin = v7.engine.joint_margin_report(
        tube[v7.v6_utils.atlas.BETA_COLS].to_numpy(dtype=float),
        domain=v7._domain(),
        at_bound_tolerance_deg=v7._margin_policy().at_bound_tolerance_deg,
    )
    margin_gate = v7.engine.evaluate_joint_margin_gate(
        margin,
        policy=v7._margin_policy(),
    )
    replay = v7.replay_tube_quality_gate(
        source,
        source_report_sha256=v7.file_sha256(report_path),
        artifact_sha256=v7.file_sha256(artifact),
        centerline_sha256=v7.file_sha256(centerline),
        recomputed_geometry=geometric,
        recomputed_margin_gate={**margin, **margin_gate},
        expected_cut_indices=v7.parse_int_csv(v7_args.cut_indices),
    )
    replay.update(
        {
            "replay_strategy_version": REPLAY_STRATEGY_VERSION,
            "source_report_path": str(report_path.resolve()),
            "source_artifact_path": str(artifact),
            "source_centerline_path": str(centerline),
            "tube_artifact_path": str(artifact),
            "tube_artifact_sha256": v7.file_sha256(artifact),
        }
    )
    replay["task_fingerprint"] = stable_fingerprint(
        {
            "phase": "v7_tube_gate_only_replay",
            "strategy_version": REPLAY_STRATEGY_VERSION,
            "source_report_sha256": replay["source_report_sha256"],
            "artifact_sha256": replay["tube_artifact_sha256"],
            "centerline_sha256": replay["source_centerline_sha256"],
            "protocol": v7.formal_protocol_report(v7_args)["protocol_fingerprint"],
            "tube_success_ratio_min": v7.FORMAL_TUBE_SUCCESS_RATIO_MIN,
        }
    )
    replay_report_dir.mkdir(parents=True, exist_ok=True)
    output_report = replay_report_dir / f"{v7.v6_runner.radius_slug(radius_mm)}.json"
    v7.write_json(output_report, replay)
    return tube, replay, artifact


def phase_tube_replay(
    args: argparse.Namespace,
    *,
    v7_args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    policy_args = build_v7_args(args) if v7_args is None else v7_args
    source_dir = Path(args.source_dir).resolve()
    out = Path(policy_args.out_dir) / "02_tube"
    out.mkdir(parents=True, exist_ok=True)
    radial, source_tube, source_report_hashes = _source_reports(source_dir)
    radial_frontier = float(radial["strict_geometry_rmax_mm"])
    radii = v7.materialized_dataset_radii(radial_frontier)
    source_radii = tuple(float(value) for value in source_tube.get("materialized_radii_mm", ()))
    if source_tube and (
        len(source_radii) != len(radii)
        or any(
            not any(np.isclose(radius, source, atol=1.0e-8) for source in source_radii)
            for radius in radii
        )
    ):
        raise ValueError("source tube report does not cover every materialized radius")

    tube_paths: dict[str, str] = {}
    passed_by_radius: dict[float, bool] = {}
    rows: list[dict[str, Any]] = []
    replay_reports = out / "gate_only_replay_reports"
    for radius in radii:
        try:
            _tube, quality, artifact = _replay_radius(
                policy_args,
                source_dir=source_dir,
                radius_mm=float(radius),
                replay_report_dir=replay_reports,
            )
            passed = bool(quality.get("formal_tube_label_gate_pass", False))
            reason = "passed" if passed else "replayed_tube_or_label_gate_failed"
            if passed:
                tube_paths[f"{float(radius):g}"] = str(artifact.resolve())
        except Exception as exc:  # noqa: BLE001 - preserve a fail-closed radius matrix
            quality = {}
            artifact = _quality_report_path(source_dir, float(radius)).parent / "missing"
            passed = False
            reason = f"{type(exc).__name__}: {exc}"
        passed_by_radius[float(radius)] = passed
        rows.append(
            {
                "family_id": radial.get("family_id"),
                "radius_mm": float(radius),
                "formal_tube_label_gate_pass": passed,
                "reason": reason,
                "rows": quality.get("rows"),
                "residual_p95_mm": quality.get("residual_p95_mm"),
                "residual_max_mm": quality.get("residual_max_mm"),
                "tube_success_ratio": quality.get("tube_success_ratio"),
                "tube10_beta_rms_p95_deg": quality.get("tube10_beta_rms_p95_deg"),
                "min_joint_margin_deg": quality.get("min_joint_margin_deg"),
                "tube_artifact_path": str(artifact.resolve()),
                "tube_artifact_sha256": quality.get("tube_artifact_sha256"),
                "source_report_sha256": quality.get("source_report_sha256"),
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = out / "tube_radius_summary.csv"
    summary.to_csv(summary_path, index=False)
    frontier = compute_replayed_tube_frontier(
        radial_frontier_mm=radial_frontier,
        materialized_radii_mm=radii,
        passed_by_radius=passed_by_radius,
    )
    selected_radii = tuple(float(value) for value in frontier["selected_radii_mm"])
    tube_paths = {
        key: path
        for key, path in tube_paths.items()
        if any(np.isclose(float(key), radius, atol=1.0e-8) for radius in selected_radii)
    }
    protocol = v7.formal_protocol_report(policy_args)
    report = {
        "strategy_version": v7.TUBE_STRATEGY_VERSION,
        "replay_strategy_version": REPLAY_STRATEGY_VERSION,
        **protocol,
        "gate_only_replay": True,
        "formal_tube_success_ratio_min": v7.FORMAL_TUBE_SUCCESS_RATIO_MIN,
        "source_dir": str(source_dir),
        "source_tube_summary_present": bool(source_tube),
        "source_report_hashes": source_report_hashes,
        "source_radial_task_fingerprint": str(radial.get("task_fingerprint", "")),
        "source_tube_task_fingerprint": str(source_tube.get("task_fingerprint", "")),
        "radial_task_fingerprint": str(radial.get("task_fingerprint", "")),
        "family_id": radial.get("family_id"),
        "radial_strict_rmax_mm": radial_frontier,
        "strict_geometry_rmax_mm": frontier["strict_geometry_rmax_mm"],
        "attempted_radii_mm": list(radii),
        "materialized_radii_mm": list(selected_radii),
        "all_dataset_radii_pass": frontier["all_dataset_radii_pass"],
        "tube_paths": tube_paths,
        "centerline_manifest": source_tube.get("centerline_manifest", {}),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": v7.file_sha256(summary_path),
    }
    report.update(
        v7.tube_frontier_gate_decision(
            formal_protocol_gate_pass=bool(protocol["formal_protocol_gate_pass"]),
            radial_strict_rmax_mm=radial_frontier,
            tube_strict_rmax_mm=frontier["strict_geometry_rmax_mm"],
            all_dataset_radii_pass=bool(frontier["all_dataset_radii_pass"]),
        )
    )
    report["task_fingerprint"] = stable_fingerprint(
        {
            "phase": "v7_tube_gate_only_replay_summary",
            "strategy_version": REPLAY_STRATEGY_VERSION,
            "protocol": report["protocol_fingerprint"],
            "source_report_hashes": source_report_hashes,
            "radii": [
                {
                    "radius_mm": row["radius_mm"],
                    "pass": row["formal_tube_label_gate_pass"],
                    "artifact_sha256": row["tube_artifact_sha256"],
                    "source_report_sha256": row["source_report_sha256"],
                }
                for row in rows
            ],
        }
    )
    v7.write_json(out / "tube_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    v7_args = build_v7_args(args)
    tube = phase_tube_replay(args, v7_args=v7_args)
    dataset = v7.phase_dataset(v7_args, tube_report=tube)
    summary = {
        "strategy_version": REPLAY_STRATEGY_VERSION,
        "gate_only_replay": True,
        "source_dir": str(Path(args.source_dir).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()),
        "formal_tube_success_ratio_min": v7.FORMAL_TUBE_SUCCESS_RATIO_MIN,
        "strict_radial_rmax_mm": tube.get("radial_strict_rmax_mm"),
        "strict_tube_rmax_mm": tube.get("strict_geometry_rmax_mm"),
        "target_105_achieved": bool(tube.get("target_105_achieved", False)),
        "validation_radius_mm": (dataset.get("holdout") or {}).get("validation_radius_mm"),
        "test_radius_mm": (dataset.get("holdout") or {}).get("test_radius_mm"),
        "formal_tube_gate_pass": bool(tube.get("formal_tube_gate_pass", False)),
        "formal_dataset_gate_pass": bool(dataset.get("formal_dataset_gate_pass", False)),
        "model_training_authorized": bool(dataset.get("formal_dataset_gate_pass", False)),
        "tube_task_fingerprint": str(tube.get("task_fingerprint", "")),
        "dataset_task_fingerprint": str(dataset.get("task_fingerprint", "")),
    }
    summary_dir = Path(v7_args.out_dir) / "04_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    v7.write_json(summary_dir / "summary_report.json", summary)
    report = {
        "mode": "true_ellipse_standard_domain_v7_tube_gate_replay",
        "tube": tube,
        "dataset": dataset,
        "summary": summary,
    }
    v7.write_json(Path(v7_args.out_dir) / "run_report.json", report)
    return report


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, default=v7._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
