from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    analysis_dir = REPO_ROOT / "scripts" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    path = analysis_dir / "run_true_ellipse_standard_domain_tube_gate_replay_v7.py"
    spec = importlib.util.spec_from_file_location(
        "run_true_ellipse_standard_domain_tube_gate_replay_v7", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_replay_cli_isolated_from_source_and_uses_current_registered_gate(tmp_path: Path) -> None:
    mod = _load_module()
    source = tmp_path / "source"
    output = tmp_path / "replay"

    args = mod.parse_args(["--source-dir", str(source), "--out-dir", str(output)])
    v7_args = mod.build_v7_args(args)

    assert args.source_dir == source
    assert args.out_dir == output
    assert args.source_dir != args.out_dir
    assert mod.v7.FORMAL_TUBE_SUCCESS_RATIO_MIN == 0.99
    assert v7_args.out_dir == output
    assert mod.v7.formal_protocol_report(v7_args)["formal_protocol_gate_pass"] is True


def test_replayed_frontier_selects_highest_complete_registered_prefix() -> None:
    mod = _load_module()
    radii = (75.0, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0, 101.25, 102.5)
    all_pass = {radius: True for radius in radii}

    complete = mod.compute_replayed_tube_frontier(
        radial_frontier_mm=102.5,
        materialized_radii_mm=radii,
        passed_by_radius=all_pass,
    )
    failed_anchor = mod.compute_replayed_tube_frontier(
        radial_frontier_mm=102.5,
        materialized_radii_mm=radii,
        passed_by_radius={**all_pass, 101.25: False},
    )

    assert complete["strict_geometry_rmax_mm"] == 102.5
    assert complete["all_dataset_radii_pass"] is True
    assert failed_anchor["strict_geometry_rmax_mm"] == 100.0
    assert failed_anchor["all_dataset_radii_pass"] is True
    assert failed_anchor["selected_radii_mm"][-1] == 100.0


def test_source_radial_evidence_remains_replayable_when_outer_tube_summary_is_absent(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    source = tmp_path / "source"
    radial_dir = source / "01_radial"
    radial_dir.mkdir(parents=True)
    centerline = radial_dir / "centerline.parquet"
    centerline.write_bytes(b"hash-bound-centerline")
    radial = {
        "downstream_radial_admission_gate_pass": True,
        "strict_geometry_rmax_mm": 102.5,
        "task_fingerprint": "radial-task",
        "path_manifest": {
            "100": {
                "path": str(centerline),
                "sha256": mod.v7.file_sha256(centerline),
            }
        },
    }
    radial_report = radial_dir / "radial_report.json"
    radial_report.write_text(json.dumps(radial), encoding="utf-8")

    loaded_radial, loaded_tube, hashes = mod._source_reports(source)

    assert loaded_radial["task_fingerprint"] == "radial-task"
    assert loaded_tube == {}
    assert set(hashes) == {"radial"}
