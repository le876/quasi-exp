from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/analysis"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_bacra_v12 as v12
from quasi_exp.provenance import (
    ARTIFACT_MANIFEST_TYPE,
    FAMILY_DISPOSITION_TYPE,
    SOURCE_FIXED_POINT_TYPE,
    FamilyDispositionError,
    FileRecord,
    SourceLockError,
    assert_final_sealed_family_eligible,
    collect_tree,
    load_family_disposition,
    load_json_object,
    preflight_source_fixed_point,
    sha256_file,
    tree_summary,
    verify_artifact_tree,
    verify_protocol_source_snapshot,
    write_protocol_source_snapshot,
)


PROVENANCE_ROOT = ROOT / "docs/provenance/v12_12_retry3"
ARTIFACT_ID = (
    "branch_aware_canonical_region_atlas_v12_12_"
    "independent_gold_set_geometry_holdout_retry3"
)
ARTIFACT_ROOT = v12.project_root_from(ROOT) / "runs" / ARTIFACT_ID
RECOVERED_RUNNER_SHA256 = (
    "7401473aa96939e7fd67ce19a9b4a8ee5bb0bf1b57673cb25afc39ee80f402f8"
)
FINAL_STAGE_RUNNER_SHA256 = (
    "6b5ad79c648b3db93efbfef7ac0ee1845a01266999ed52bede5a1275332a2130"
)


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_artifact_manifest",
        SCRIPT_DIR / "verify_artifact_manifest.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _make_artifact_manifest(root: Path) -> dict[str, Any]:
    summary = tree_summary(collect_tree(root))
    lineage = {
        stage: {
            "sha256": FINAL_STAGE_RUNNER_SHA256,
        }
        for stage in ("protocol", "teacher", "student", "visualize", "summary")
    }
    return {
        "schema_version": 1,
        "manifest_type": ARTIFACT_MANIFEST_TYPE,
        "artifact_id": "fixture",
        "logical_root": "runs/fixture",
        "seal": {
            "status": "retrospectively_sealed_current_state",
            "limitations": ["fixture seal"],
        },
        "tree": summary,
        "runner_lineage": lineage,
        "implementation_fixed_point": {
            "source_fixed_point_sha256": "0" * 64,
        },
        "family_disposition": {
            "sha256": "1" * 64,
        },
    }


def _make_source_repo(root: Path) -> tuple[Path, dict[str, Path]]:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "provenance-test@example.invalid")
    _git(root, "config", "user.name", "Provenance Test")
    files = {
        "config": root / "configs/formal.yaml",
        "dependency": root / "src/package/dependency.py",
        "runner": root / "scripts/run.py",
    }
    files["config"].parent.mkdir(parents=True)
    files["dependency"].parent.mkdir(parents=True)
    files["runner"].parent.mkdir(parents=True)
    files["config"].write_text("protocol_id: fixed\n", encoding="utf-8")
    files["dependency"].write_text("VALUE = 1\n", encoding="utf-8")
    files["runner"].write_text(
        "from package.dependency import VALUE\n", encoding="utf-8"
    )
    records = []
    for path in sorted(files.values()):
        records.append(
            FileRecord(
                path=path.relative_to(root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            ).as_dict()
        )
    manifest = root / "docs/source_fixed_point.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "manifest_type": SOURCE_FIXED_POINT_TYPE,
            "formal_config": {
                "entrypoint": "configs/formal.yaml",
            },
            "execution_sources": records,
        },
    )
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "fixture fixed point")
    files["manifest"] = manifest
    return root, files


def test_historical_runner_snapshots_have_exact_sha256() -> None:
    recovered = (
        PROVENANCE_ROOT
        / "source_snapshots"
        / (
            "run_bacra_v12_12_protocol_teacher_"
            f"{RECOVERED_RUNNER_SHA256}.py"
        )
    )
    final_stage = (
        PROVENANCE_ROOT
        / "source_snapshots"
        / (
            "run_bacra_v12_12_student_visualize_summary_"
            f"{FINAL_STAGE_RUNNER_SHA256}.py"
        )
    )
    assert sha256_file(recovered) == RECOVERED_RUNNER_SHA256
    assert sha256_file(final_stage) == FINAL_STAGE_RUNNER_SHA256


def test_formal_config_resolves_to_artifact_frozen_config() -> None:
    config_path = (
        ROOT / "configs/bacra_v12_12_1_multiresolution_teacher_retry3.yaml"
    )
    resolved = json.loads(
        json.dumps(v12.load_protocol_config(config_path, "formal"))
    )
    frozen = json.loads(
        (ARTIFACT_ROOT / "00_protocol/frozen_config.json").read_text(
            encoding="utf-8"
        )
    )
    resolved["config_path"] = Path(resolved["config_path"]).name
    frozen["config_path"] = Path(frozen["config_path"]).name
    assert resolved == frozen


def test_source_fixed_point_hashes_complete_execution_and_test_closure() -> None:
    fixed_point = load_json_object(
        PROVENANCE_ROOT / "source_fixed_point.json"
    )
    assert fixed_point["manifest_type"] == SOURCE_FIXED_POINT_TYPE
    assert fixed_point["execution_closure"] == {
        "audit_method": (
            "recursive static local-import closure from the five V12 runner "
            "entrypoints, including executed package __init__ modules, plus "
            "the full formal config chain and family disposition policy"
        ),
        "code_file_count": 38,
        "formal_config_file_count": 12,
        "policy_file_count": 1,
        "total_file_count": 51,
    }
    for group in ("execution_sources", "verification_sources"):
        paths = [item["path"] for item in fixed_point[group]]
        assert paths == sorted(paths)
        assert len(paths) == len(set(paths))
        for item in fixed_point[group]:
            path = ROOT / item["path"]
            assert path.stat().st_size == item["size_bytes"]
            assert sha256_file(path) == item["sha256"]
    assert (
        fixed_point["implementation_components"]["geometry_solver"]["sha256"]
        == "95babb5b9b20af7765b88727cfc15fa1e4187c2bdfd88955b1cdd241c1ac1970"
    )
    assert (
        fixed_point["implementation_components"]["gold_assignment"]["sha256"]
        == "183aaa1143306399b21496dcb50ba51c8db2e97ba2e22867613372bd7ec65c04"
    )
    lineage = {
        tuple(item["stages"]): item["sha256"]
        for item in fixed_point["runner_lineage"]["records"]
    }
    assert lineage[("protocol", "teacher")] == RECOVERED_RUNNER_SHA256
    assert (
        lineage[("student", "visualize", "summary")]
        == FINAL_STAGE_RUNNER_SHA256
    )
    historical = fixed_point["historical_artifact_source_manifest"]
    assert historical["scope"] == ["protocol"]
    assert "does not represent teacher-to-summary" in historical["limitation"]


def test_retry3_artifact_matches_immutable_manifest() -> None:
    manifest = load_json_object(PROVENANCE_ROOT / "artifact_manifest.json")
    assert manifest["tree"]["file_count"] == 100
    assert manifest["tree"]["total_bytes"] == 43_005_173
    assert (
        manifest["tree"]["tree_sha256"]
        == "86313dc5666c335ac87deb6f9155021fdc04cac9b2601cb5fcec7686fc63fbaa"
    )
    assert (
        manifest["implementation_fixed_point"][
            "source_fixed_point_sha256"
        ]
        == sha256_file(PROVENANCE_ROOT / "source_fixed_point.json")
    )
    assert manifest["family_disposition"]["sha256"] == sha256_file(
        PROVENANCE_ROOT / "family_disposition.json"
    )
    assert verify_artifact_tree(manifest, ARTIFACT_ROOT) == []


@pytest.mark.parametrize("mutation", ["change", "delete", "add"])
def test_artifact_verifier_rejects_any_tree_mutation(
    tmp_path: Path,
    mutation: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    nested = root / "nested/beta.bin"
    nested.parent.mkdir()
    nested.write_bytes(b"\x00\x01\x02")
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, _make_artifact_manifest(root))
    verifier = _load_verifier()
    assert (
        verifier.main(
            [
                "--manifest",
                str(manifest_path),
                "--artifact-root",
                str(root),
            ]
        )
        == 0
    )
    capsys.readouterr()
    if mutation == "change":
        (root / "alpha.txt").write_text("ALPHA\n", encoding="utf-8")
    elif mutation == "delete":
        nested.unlink()
    else:
        (root / "extra.txt").write_text("extra\n", encoding="utf-8")
    assert (
        verifier.main(
            [
                "--manifest",
                str(manifest_path),
                "--artifact-root",
                str(root),
            ]
        )
        == 1
    )


def test_artifact_verifier_rejects_invalid_schema(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, {"schema_version": 999})
    assert (
        _load_verifier().main(
            [
                "--manifest",
                str(manifest_path),
                "--artifact-root",
                str(root),
            ]
        )
        == 2
    )


def test_stress_families_are_development_boundary_by_id_and_fingerprint() -> None:
    disposition = load_family_disposition(
        PROVENANCE_ROOT / "family_disposition.json"
    )
    assert disposition["manifest_type"] == FAMILY_DISPOSITION_TYPE
    expected = {
        "independent_stress_00": (
            "6b6bd823e14509d6e1bdaa1f2a7de753a68d240bd25a2cffba7b11500efe44fc"
        ),
        "independent_stress_01": (
            "0c2ab37ece57c939cf5acd52529ff354a5e11e8fcb9b50acbe752deb7c82d6ff"
        ),
        "independent_stress_02": (
            "6c79c4834ba6ad7a2c2c623075367587ea78188156b30e3a9c8d47ef606b56b3"
        ),
        "independent_stress_03": (
            "1bdd2157f530eecac6ae7ff934ece6111aee8ca93e6fc3572f7343d4064f5459"
        ),
    }
    entries = {
        item["family_id"]: item
        for item in disposition["development_boundary_families"]
    }
    assert {
        family_id: item["family_fingerprint"]
        for family_id, item in entries.items()
    } == expected
    assert all(
        item["final_sealed_test_eligible"] is False
        for item in entries.values()
    )
    for family_id, fingerprint in expected.items():
        with pytest.raises(FamilyDispositionError, match="family_id"):
            assert_final_sealed_family_eligible(
                family_id, sha256(family_id.encode()).hexdigest(), disposition
            )
        with pytest.raises(
            FamilyDispositionError, match="family_fingerprint"
        ):
            assert_final_sealed_family_eligible(
                f"renamed_{family_id}", fingerprint, disposition
            )


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_source_preflight_rejects_dirty_or_untracked_source(
    tmp_path: Path,
    dirty_kind: str,
) -> None:
    root, files = _make_source_repo(tmp_path / "source")
    if dirty_kind == "tracked":
        files["runner"].write_text("VALUE = 2\n", encoding="utf-8")
    else:
        (root / "untracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(SourceLockError, match="worktree is dirty"):
        preflight_source_fixed_point(
            root,
            files["manifest"],
            requested_config=files["config"],
        )


def test_protocol_snapshot_copies_full_locked_source_tree(
    tmp_path: Path,
) -> None:
    root, files = _make_source_repo(tmp_path / "source")
    state = preflight_source_fixed_point(
        root,
        files["manifest"],
        requested_config=files["config"],
    )
    output = tmp_path / "artifact"
    lock = write_protocol_source_snapshot(state, output)
    assert lock["git_sha"] == state.git_sha
    assert lock["source_fixed_point_sha256"] == state.manifest_sha256
    assert verify_protocol_source_snapshot(state, output) == lock
    copied_paths = {
        path.relative_to(
            output / "00_protocol/source_snapshot"
        ).as_posix()
        for path in (output / "00_protocol/source_snapshot").rglob("*")
        if path.is_file() and path.name != "source_lock.json"
    }
    assert copied_paths == {
        state.manifest_relative_path,
        *(record.path for record in state.execution_sources),
    }


@pytest.mark.parametrize(
    "drift",
    ["runner", "dependency", "config", "git_sha"],
)
def test_later_stage_fails_closed_before_write_on_source_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    root, files = _make_source_repo(tmp_path / "source")
    state = preflight_source_fixed_point(
        root,
        files["manifest"],
        requested_config=files["config"],
    )
    output = tmp_path / "artifact"
    write_protocol_source_snapshot(state, output)
    later_result = output / "01_teacher_reference/result.json"
    if drift == "git_sha":
        marker = root / "commit_marker.txt"
        marker.write_text("new commit\n", encoding="utf-8")
        _git(root, "add", "commit_marker.txt")
        _git(root, "commit", "-q", "-m", "switch Git SHA")
        changed_state = preflight_source_fixed_point(
            root,
            files["manifest"],
            requested_config=files["config"],
        )
        with pytest.raises(SourceLockError, match="Git SHA"):
            verify_protocol_source_snapshot(changed_state, output)
    else:
        files[drift].write_text(
            files[drift].read_text(encoding="utf-8") + "# drift\n",
            encoding="utf-8",
        )
        with pytest.raises(SourceLockError):
            preflight_source_fixed_point(
                root,
                files["manifest"],
                requested_config=files["config"],
            )
    assert not later_result.exists()
