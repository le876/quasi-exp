from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/spec/verify_spec_system.py"


def _module():
    spec = importlib.util.spec_from_file_location("verify_spec_system", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_binding(
    root: Path,
    experiment_id: str,
    *,
    upstream_experiments: list[str] | None = None,
):
    for directory in (
        "configs",
        "docs",
        "scripts/analysis",
        "scripts/pipelines",
        "tests",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    protocol = f"docs/{experiment_id}.md"
    config = f"configs/{experiment_id}.yaml"
    runner = f"scripts/analysis/{experiment_id}.py"
    launcher = f"scripts/pipelines/{experiment_id}.sh"
    test = f"tests/test_{experiment_id}.py"
    (root / protocol).write_text("# Protocol\n", encoding="utf-8")
    (root / config).write_text(
        f"experiment_id: {experiment_id}\n"
        f"output_root: runs/{experiment_id}\n"
        "sources:\n"
        f"  protocol: {protocol}\n",
        encoding="utf-8",
    )
    for relative in (runner, launcher, test):
        (root / relative).write_text("\n", encoding="utf-8")
    entry = {
        "role": "fixture",
        "scientific_source_fixed_point": "a" * 40,
        "protocol_sources": [protocol],
        "config": config,
        "runner": runner,
        "launcher": launcher,
        "tests": {"self_contained": [test], "artifact_bound": []},
        "upstream_experiments": upstream_experiments or [],
    }
    return entry


def _write_registry(root: Path, experiments: dict[str, object]) -> None:
    module = _module()
    (root / "spec").mkdir(parents=True, exist_ok=True)
    (root / "spec/registry.yaml").write_text(
        module.yaml.safe_dump(
            {
                "schema_version": 3,
                "experiments": experiments,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_current_state(root: Path, **overrides: object) -> dict[str, object]:
    module = _module()
    metadata: dict[str, object] = {
        "schema_version": 3,
        "observed_at": "2026-08-21T00:00:00+08:00",
        "current_experiment": "current",
        "current_attempt": "retry5",
        "run_root": "runs/current/retry5",
        "runtime_task": "task",
        "runtime_state_locator": "runs/maintenance/task.state",
        "runtime_log_locator": "runs/maintenance/task.log",
        "summary_artifact_locator": None,
        "current_consultation_locator": None,
    }
    metadata.update(overrides)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs/current-state.md").write_text(
        "---\n"
        + module.yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
        + "---\n\n# Current pointer\n",
        encoding="utf-8",
    )
    return metadata


def _note_text(headings: tuple[str, ...]) -> str:
    return "\n\n".join(heading + "\n\nEvidence." for heading in headings) + "\n"


def _write_note(
    path: Path,
    *,
    lifecycle: str,
    rejection_reason: str | None = None,
    archived_date: str = "2026-08-20",
    include_verification: bool = False,
) -> None:
    module = _module()
    status = "implemented" if lifecycle == "archived" else lifecycle
    if lifecycle == "rejected":
        status_line = "Status: rejected — " + (
            "Evidence rejected the proposal." if rejection_reason is None else rejection_reason
        )
    else:
        status_line = "Status: " + status
    if lifecycle in {"implemented", "archived"}:
        headings = module.IMPLEMENTED_HEADINGS
    elif lifecycle == "proposed":
        headings = module.PROPOSED_HEADINGS
    else:
        headings = module.REJECTED_HEADINGS
    if include_verification:
        headings = (*headings, "## Verification")
    archive_line = f"\nArchived: {archived_date}" if lifecycle == "archived" else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Agent Note: Fixture decision\n\n"
        + status_line
        + archive_line
        + "\n\n"
        + _note_text(headings),
        encoding="utf-8",
    )


def _setup_notes(root: Path) -> Path:
    notes = root / ".agents/notes"
    notes.mkdir(parents=True)
    (notes / "README.md").write_text("# Agent Notes\n", encoding="utf-8")
    _write_note(
        notes / "implemented/2026-08-18-base.md",
        lifecycle="implemented",
    )
    (notes / "archived").mkdir()
    return notes


def _write_skill(
    root: Path,
    name: str,
    *,
    include_openai: bool = True,
    include_policy: bool = True,
    policy_value: str = "true",
    add_reference: bool = False,
) -> Path:
    skill = root / ".agents/skills" / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Validate repository skill discovery and metadata.\n"
        "---\n\n"
        "# Test Skill\n\nPerform the requested validation.\n",
        encoding="utf-8",
    )
    if include_openai:
        (skill / "agents").mkdir()
        metadata = (
            "interface:\n"
            '  display_name: "Test Skill"\n'
            '  short_description: "Validate repository skill metadata"\n'
            f'  default_prompt: "Use ${name} to validate this fixture."\n'
        )
        if include_policy:
            metadata += "policy:\n" f"  allow_implicit_invocation: {policy_value}\n"
        (skill / "agents/openai.yaml").write_text(metadata, encoding="utf-8")
    if add_reference:
        (skill / "references").mkdir()
        (skill / "references/example.md").write_text("# Reference\n", encoding="utf-8")
    return skill


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _init_git_fixture(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Spec Test")
    _git(root, "config", "user.email", "spec-test@example.invalid")


def _commit_fixture(root: Path) -> str:
    _init_git_fixture(root)
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "scientific source")
    return _git(root, "rev-parse", "HEAD")


def _create_publication_pair(
    root: Path,
    *,
    mutation: str | None = None,
    source_message: str = "scientific source without publication metadata",
    public_message: str = "public snapshot without source metadata",
) -> tuple[str, str]:
    module = _module()
    _init_git_fixture(root)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text("# fixture\n", encoding="utf-8")
    config_path = root / "configs/released.yaml"
    config_path.write_text("value: source\n", encoding="utf-8")
    (root / "docs/not-public.md").write_text("# Private note\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", source_message)
    source_sha = _git(root, "rev-parse", "HEAD")

    (root / "docs/not-public.md").unlink()
    if mutation == "missing":
        config_path.unlink()
    elif mutation == "extra":
        (root / "outside.txt").write_text("not allowed\n", encoding="utf-8")
    elif mutation == "content":
        config_path.write_text("value: public\n", encoding="utf-8")
    _git(root, "add", "-A")
    if mutation == "mode":
        _git(root, "update-index", "--chmod=+x", "configs/released.yaml")
    elif mutation == "type":
        _git(
            root,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{source_sha},configs/released.yaml",
        )
    _git(root, "commit", "-q", "--allow-empty", "-m", public_message)
    public_sha = _git(root, "rev-parse", "HEAD")
    assert source_sha != public_sha
    assert module.SHA_RE.fullmatch(source_sha)
    assert module.SHA_RE.fullmatch(public_sha)
    return source_sha, public_sha


def _write_publication_manifest(
    root: Path,
    source_sha: str,
    public_sha: str,
    **overrides: object,
) -> Path:
    module = _module()
    report = module.VerificationReport()
    source_tree = module._git_tree_entries(
        root,
        source_sha,
        label="fixture.source",
        report=report,
    )
    if source_tree is None:
        file_count = 0
        digest = "0" * 64
    else:
        filtered_source = {
            path: entry
            for path, entry in source_tree.items()
            if module._publication_path_is_included(path)
        }
        inventory = module._publication_inventory(
            root,
            filtered_source,
            label="fixture.source",
            report=report,
        )
        assert inventory is not None, report.errors
        file_count = inventory.file_count
        digest = inventory.digest
    value: dict[str, object] = {
        "schema_version": 1,
        "source_sha": source_sha,
        "public_sha": public_sha,
        "filter_policy": module.PUBLICATION_POLICY,
        "inventory_algorithm": module.PUBLICATION_INVENTORY_ALGORITHM,
        "file_count": file_count,
        "inventory_sha256": digest,
    }
    value.update(overrides)
    path = root / "spec/publication-manifests/published.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        module.yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_release_map(
    root: Path,
    unpublished: list[object],
    *,
    scientific_source_sha: str | None = None,
    public_release_sha: str | None = None,
) -> tuple[str, str]:
    module = _module()
    if scientific_source_sha is None or public_release_sha is None:
        scientific_source_sha, public_release_sha = _create_publication_pair(root)
    manifest_path = _write_publication_manifest(
        root,
        scientific_source_sha,
        public_release_sha,
    )
    (root / "spec").mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": 3,
        "public_repository": "https://github.com/example/quasi-exp",
        "current_public_release_id": "published",
        "verified_at": "2026-08-21",
        "mappings": [
            {
                "id": "published",
                "status": "verified",
                "scientific_source_sha": scientific_source_sha,
                "public_release_sha": public_release_sha,
                "evidence": {
                    "type": module.PUBLICATION_EVIDENCE_TYPE,
                    "manifest": manifest_path.relative_to(root).as_posix(),
                    "commit_url": "https://github.com/example/quasi-exp/commit/"
                    + public_release_sha,
                },
            }
        ],
        "unpublished_scientific_fixed_points": unpublished,
    }
    (root / "spec/release-map.yaml").write_text(
        module.yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return scientific_source_sha, public_release_sha


def test_repository_governance_structure_is_valid() -> None:
    module = _module()
    report = module.verify_repository(ROOT)
    assert report.errors == (), [item.message for item in report.errors]


def test_repository_path_escape_is_rejected(tmp_path: Path) -> None:
    module = _module()
    report = module.VerificationReport()
    target = module.resolve_project_path(
        tmp_path,
        "../outside.md",
        report,
        "fixture.path",
        require_exists=False,
    )
    assert target is None
    assert [item.code for item in report.errors] == ["path.escape"]


def test_registry_v3_needs_no_active_field_and_allows_zero_upstream(tmp_path: Path) -> None:
    module = _module()
    experiments = {"definition": _write_binding(tmp_path, "definition")}
    _write_registry(tmp_path, experiments)
    report = module.VerificationReport()
    registry = module.validate_registry(tmp_path, report)
    assert registry is not None
    assert "active_experiment" not in registry
    assert report.errors == ()


def test_registry_v3_allows_one_and_multiple_upstreams(tmp_path: Path) -> None:
    module = _module()
    experiments = {
        "source": _write_binding(tmp_path, "source"),
        "middle": _write_binding(
            tmp_path,
            "middle",
            upstream_experiments=["source"],
        ),
        "downstream": _write_binding(
            tmp_path,
            "downstream",
            upstream_experiments=["source", "middle"],
        ),
    }
    _write_registry(tmp_path, experiments)
    report = module.VerificationReport()
    module.validate_registry(tmp_path, report)
    assert report.errors == ()


def test_registry_rejects_unknown_upstream(tmp_path: Path) -> None:
    module = _module()
    entry = _write_binding(tmp_path, "example", upstream_experiments=["missing"])
    _write_registry(tmp_path, {"example": entry})
    report = module.VerificationReport()
    module.validate_registry(tmp_path, report)
    assert "registry.upstream" in {item.code for item in report.errors}


def test_registry_rejects_upstream_cycle(tmp_path: Path) -> None:
    module = _module()
    experiments = {
        "a": _write_binding(tmp_path, "a", upstream_experiments=["b"]),
        "b": _write_binding(tmp_path, "b", upstream_experiments=["a"]),
    }
    _write_registry(tmp_path, experiments)
    report = module.VerificationReport()
    module.validate_registry(tmp_path, report)
    assert "registry.upstream_cycle" in {item.code for item in report.errors}


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("output_root", "runs/example"),
        ("result_source", "sealed_run_artifacts"),
        ("binding_status", "registered"),
        ("gate_pass", True),
        ("authorization", {"allowed": True}),
    ],
)
def test_registry_rejects_runtime_or_result_fields(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    module = _module()
    entry = _write_binding(tmp_path, "example")
    entry[field_name] = value
    report = module.VerificationReport()
    module.validate_experiment(tmp_path, "example", entry, report)
    assert "registry.schema" in {item.code for item in report.errors}


def test_registry_rejects_top_level_active_experiment(tmp_path: Path) -> None:
    module = _module()
    _write_registry(tmp_path, {"example": _write_binding(tmp_path, "example")})
    registry_path = tmp_path / "spec/registry.yaml"
    registry = module.yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["active_experiment"] = "example"
    registry_path.write_text(
        module.yaml.safe_dump(registry, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_registry(tmp_path, report)
    assert "registry.schema" in {item.code for item in report.errors}


def test_reverse_protocol_inventory_gap_is_scientific_advisory(tmp_path: Path) -> None:
    module = _module()
    entry = _write_binding(tmp_path, "active")
    extra = "docs/extra.md"
    (tmp_path / extra).write_text("# Extra\n", encoding="utf-8")
    entry["protocol_sources"].append(extra)
    entry["scientific_source_fixed_point"] = _commit_fixture(tmp_path)
    registry = {"schema_version": 3, "experiments": {"active": entry}}
    report = module.VerificationReport()
    module.validate_fixed_point_bindings(tmp_path, registry, report)
    assert report.errors == ()
    assert "scientific.protocol_inventory" in {item.code for item in report.warnings}


def test_registry_allows_direct_runner_and_external_binding_metadata(tmp_path: Path) -> None:
    module = _module()
    entry = _write_binding(tmp_path, "direct")
    entry["launcher"] = None
    (tmp_path / entry["config"]).write_text("robot:\n  k: 3\n", encoding="utf-8")
    report = module.VerificationReport()
    module.validate_experiment(tmp_path, "direct", entry, report)
    assert report.errors == ()
    assert report.warnings == ()


def test_scientific_source_must_be_a_real_git_commit() -> None:
    module = _module()
    registry = module.yaml.safe_load((ROOT / "spec/registry.yaml").read_text(encoding="utf-8"))
    experiment_id = next(iter(registry["experiments"]))
    registry["experiments"][experiment_id]["scientific_source_fixed_point"] = "f" * 40
    report = module.VerificationReport()
    module.validate_fixed_point_bindings(ROOT, registry, report)
    assert "git.source_missing" in {item.code for item in report.errors}


def test_historical_binding_is_valid_from_source_commit_after_worktree_drift(
    tmp_path: Path,
) -> None:
    module = _module()
    entry = _write_binding(tmp_path, "historical")
    entry["scientific_source_fixed_point"] = _commit_fixture(tmp_path)
    (tmp_path / entry["config"]).write_text("not: the source config\n", encoding="utf-8")
    (tmp_path / entry["runner"]).unlink()
    registry = {"schema_version": 3, "experiments": {"historical": entry}}
    report = module.VerificationReport()
    module.validate_experiment(tmp_path, "historical", entry, report)
    module.validate_fixed_point_bindings(tmp_path, registry, report)
    assert report.errors == ()


def test_source_commit_must_contain_every_declared_binding(tmp_path: Path) -> None:
    module = _module()
    entry = _write_binding(tmp_path, "historical")
    entry["scientific_source_fixed_point"] = _commit_fixture(tmp_path)
    entry["runner"] = "scripts/analysis/missing.py"
    report = module.VerificationReport()
    module.validate_fixed_point_bindings(
        tmp_path,
        {"schema_version": 3, "experiments": {"historical": entry}},
        report,
    )
    assert "git.binding_missing" in {item.code for item in report.errors}


@pytest.mark.parametrize(
    ("config_text", "expected"),
    [
        (
            "experiment_id: wrong\nsources:\n  protocol: docs/historical.md\n",
            "registry.experiment_id",
        ),
        (
            "experiment_id: historical\nsources:\n  protocol: docs/extra.md\n",
            "registry.protocol_binding",
        ),
        (
            "experiment_id: historical\n"
            "sources:\n"
            "  protocol: docs/historical.md\n"
            "  helper: configs/missing.yaml\n",
            "git.source_inventory_missing",
        ),
    ],
)
def test_source_bound_config_identity_and_protocol_inventory_are_enforced(
    tmp_path: Path,
    config_text: str,
    expected: str,
) -> None:
    module = _module()
    entry = _write_binding(tmp_path, "historical")
    (tmp_path / "docs/extra.md").write_text("# Extra\n", encoding="utf-8")
    (tmp_path / entry["config"]).write_text(config_text, encoding="utf-8")
    entry["scientific_source_fixed_point"] = _commit_fixture(tmp_path)
    report = module.VerificationReport()
    module.validate_fixed_point_bindings(
        tmp_path,
        {"schema_version": 3, "experiments": {"historical": entry}},
        report,
    )
    assert expected in {item.code for item in report.errors}


def test_release_map_v3_covers_sources_without_experiment_membership(tmp_path: Path) -> None:
    module = _module()
    source_sha, _public_sha = _write_release_map(tmp_path, [])
    registry = {
        "experiments": {
            "first": {"scientific_source_fixed_point": source_sha},
            "new_definition": {"scientific_source_fixed_point": source_sha},
        }
    }
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, registry, report)
    assert report.errors == ()


def test_release_map_v3_requires_source_coverage(tmp_path: Path) -> None:
    module = _module()
    registry = {"experiments": {"missing": {"scientific_source_fixed_point": "a" * 40}}}
    _write_release_map(tmp_path, [])
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, registry, report)
    assert "release.coverage" in {item.code for item in report.errors}


@pytest.mark.parametrize(
    ("unpublished", "expected"),
    [
        (["invalid"], "release.source_sha"),
        (["a" * 40, "a" * 40], "release.source_sha"),
    ],
)
def test_release_map_v3_rejects_invalid_or_duplicate_unpublished_sources(
    tmp_path: Path,
    unpublished: list[object],
    expected: str,
) -> None:
    module = _module()
    _write_release_map(tmp_path, unpublished)
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, {"experiments": {}}, report)
    assert expected in {item.code for item in report.errors}


def test_release_map_v3_rejects_released_unpublished_overlap(tmp_path: Path) -> None:
    module = _module()
    source_sha, _public_sha = _write_release_map(tmp_path, [])
    path = tmp_path / "spec/release-map.yaml"
    value = module.yaml.safe_load(path.read_text(encoding="utf-8"))
    value["unpublished_scientific_fixed_points"] = [source_sha]
    path.write_text(
        module.yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, {"experiments": {}}, report)
    assert "release.overlap" in {item.code for item in report.errors}


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        ("scientific_source_sha", "release.source_sha"),
        ("public_release_sha", "release.public_sha"),
    ],
)
def test_release_map_v3_rejects_duplicate_released_shas(
    tmp_path: Path,
    field_name: str,
    expected: str,
) -> None:
    module = _module()
    _write_release_map(tmp_path, [])
    path = tmp_path / "spec/release-map.yaml"
    value = module.yaml.safe_load(path.read_text(encoding="utf-8"))
    duplicate = dict(value["mappings"][0])
    duplicate["id"] = "duplicate"
    if field_name == "scientific_source_sha":
        duplicate["public_release_sha"] = "d" * 40
    else:
        duplicate["scientific_source_sha"] = "d" * 40
    duplicate["evidence"] = {
        "type": module.PUBLICATION_EVIDENCE_TYPE,
        "manifest": value["mappings"][0]["evidence"]["manifest"],
        "commit_url": value["public_repository"]
        + "/commit/"
        + duplicate["public_release_sha"],
    }
    value["mappings"].append(duplicate)
    path.write_text(
        module.yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, {"experiments": {}}, report)
    assert expected in {item.code for item in report.errors}


def test_publication_manifest_proves_filtered_tree_without_commit_message_identity(
    tmp_path: Path,
) -> None:
    module = _module()
    source_sha, public_sha = _create_publication_pair(
        tmp_path,
        source_message="source message names the wrong public release",
        public_message="public message names no scientific source",
    )
    _write_release_map(
        tmp_path,
        [],
        scientific_source_sha=source_sha,
        public_release_sha=public_sha,
    )
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, {"experiments": {}}, report)
    assert report.errors == ()


@pytest.mark.parametrize("missing_kind", ["source", "public"])
def test_publication_manifest_requires_both_commits(
    tmp_path: Path,
    missing_kind: str,
) -> None:
    module = _module()
    source_sha, public_sha = _create_publication_pair(tmp_path)
    _write_release_map(
        tmp_path,
        [],
        scientific_source_sha=source_sha,
        public_release_sha=public_sha,
    )
    missing_sha = "d" * 40
    release_path = tmp_path / "spec/release-map.yaml"
    release_map = module.yaml.safe_load(release_path.read_text(encoding="utf-8"))
    manifest_path = tmp_path / release_map["mappings"][0]["evidence"]["manifest"]
    manifest = module.yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    field = "scientific_source_sha" if missing_kind == "source" else "public_release_sha"
    manifest_field = "source_sha" if missing_kind == "source" else "public_sha"
    release_map["mappings"][0][field] = missing_sha
    manifest[manifest_field] = missing_sha
    release_map["mappings"][0]["evidence"]["commit_url"] = (
        release_map["public_repository"]
        + "/commit/"
        + release_map["mappings"][0]["public_release_sha"]
    )
    release_path.write_text(
        module.yaml.safe_dump(release_map, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest_path.write_text(
        module.yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, {"experiments": {}}, report)
    assert "release.commit_missing" in {item.code for item in report.errors}
    if missing_kind == "public":
        assert any("git fetch origin main" in item.message for item in report.errors)


def test_publication_manifest_must_bind_release_map_shas(tmp_path: Path) -> None:
    module = _module()
    _write_release_map(tmp_path, [])
    manifest_path = tmp_path / "spec/publication-manifests/published.yaml"
    manifest = module.yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["source_sha"] = "d" * 40
    manifest_path.write_text(
        module.yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, {"experiments": {}}, report)
    assert "release.manifest_binding" in {item.code for item in report.errors}


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", "release.tree"),
        ("extra", "release.tree"),
        ("content", "release.tree"),
        ("mode", "release.tree"),
        ("type", "release.entry_type"),
    ],
)
def test_publication_manifest_rejects_tree_differences(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    module = _module()
    source_sha, public_sha = _create_publication_pair(tmp_path, mutation=mutation)
    _write_release_map(
        tmp_path,
        [],
        scientific_source_sha=source_sha,
        public_release_sha=public_sha,
    )
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, {"experiments": {}}, report)
    assert expected in {item.code for item in report.errors}


@pytest.mark.parametrize(
    ("field_name", "replacement", "expected"),
    [
        ("filter_policy", "unknown", "release.policy"),
        ("inventory_algorithm", "unknown", "release.inventory"),
        ("file_count", 999, "release.inventory"),
        ("inventory_sha256", "0" * 64, "release.inventory"),
    ],
)
def test_publication_manifest_rejects_policy_or_inventory_drift(
    tmp_path: Path,
    field_name: str,
    replacement: object,
    expected: str,
) -> None:
    module = _module()
    _write_release_map(tmp_path, [])
    manifest_path = tmp_path / "spec/publication-manifests/published.yaml"
    manifest = module.yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest[field_name] = replacement
    manifest_path.write_text(
        module.yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, {"experiments": {}}, report)
    assert expected in {item.code for item in report.errors}


def test_publication_commit_url_must_match_public_sha(tmp_path: Path) -> None:
    module = _module()
    _write_release_map(tmp_path, [])
    path = tmp_path / "spec/release-map.yaml"
    value = module.yaml.safe_load(path.read_text(encoding="utf-8"))
    value["mappings"][0]["evidence"]["commit_url"] = (
        value["public_repository"] + "/commit/" + "d" * 40
    )
    path.write_text(
        module.yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, {"experiments": {}}, report)
    assert "release.evidence" in {item.code for item in report.errors}


def test_release_map_rejects_commit_message_evidence_fallback(tmp_path: Path) -> None:
    module = _module()
    _write_release_map(tmp_path, [])
    path = tmp_path / "spec/release-map.yaml"
    value = module.yaml.safe_load(path.read_text(encoding="utf-8"))
    value["mappings"][0]["evidence"]["type"] = "github_commit_message"
    path.write_text(
        module.yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_release_map(tmp_path, {"experiments": {}}, report)
    assert "release.evidence" in {item.code for item in report.errors}


def test_publication_manifest_is_a_persistence_path(tmp_path: Path) -> None:
    _write_release_map(tmp_path, [])
    module = _module()
    assert "spec/publication-manifests/published.yaml" in module.persistence_paths(
        tmp_path,
        None,
    )


def test_current_state_selects_registered_experiment_and_attempt(tmp_path: Path) -> None:
    module = _module()
    _write_current_state(tmp_path)
    registry = {"experiments": {"current": {}}}
    report = module.VerificationReport()
    module.validate_current_state(tmp_path, registry, report)
    assert report.errors == ()


@pytest.mark.parametrize("consultation", [None, "docs/21-pro提问-example.md"])
def test_current_state_allows_null_or_existing_consultation(
    tmp_path: Path,
    consultation: str | None,
) -> None:
    module = _module()
    if consultation is not None:
        path = tmp_path / consultation
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Question\n", encoding="utf-8")
    _write_current_state(tmp_path, current_consultation_locator=consultation)
    report = module.VerificationReport()
    module.validate_current_state(tmp_path, {"experiments": {"current": {}}}, report)
    assert report.errors == ()


@pytest.mark.parametrize(
    ("consultation", "expected"),
    [
        ("../outside.md", "path.escape"),
        ("docs/missing.md", "path.missing"),
    ],
)
def test_current_state_rejects_unsafe_or_missing_consultation(
    tmp_path: Path,
    consultation: str,
    expected: str,
) -> None:
    module = _module()
    _write_current_state(tmp_path, current_consultation_locator=consultation)
    report = module.VerificationReport()
    module.validate_current_state(tmp_path, {"experiments": {"current": {}}}, report)
    assert expected in {item.code for item in report.errors}


@pytest.mark.parametrize("field_name", ["runtime_status", "gate_pass", "authorization"])
def test_current_state_rejects_result_cache_fields(
    tmp_path: Path,
    field_name: str,
) -> None:
    module = _module()
    _write_current_state(tmp_path, **{field_name: True})
    report = module.VerificationReport()
    module.validate_current_state(tmp_path, {"experiments": {"current": {}}}, report)
    codes = {item.code for item in report.errors}
    assert "state.schema" in codes
    assert "state.result_field" in codes


def test_budget_checks_only_manifest_listed_standing_docs(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "scripts/spec").mkdir(parents=True)
    (tmp_path / ".agents/skills/example").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "AGENTS.md").write_text("x" * 11, encoding="utf-8")
    (tmp_path / "src/AGENTS.md").write_text("x" * 11, encoding="utf-8")
    (tmp_path / ".agents/skills/example/SKILL.md").write_text(
        "---\nname: example\ndescription: " + "d" * 12 + "\n---\n" + "s" * 11,
        encoding="utf-8",
    )
    (tmp_path / "scripts/spec/doc-budgets.yaml").write_text(
        "schema_version: 2\n"
        "unit: unicode_characters\n"
        "files:\n"
        "  AGENTS.md: 10\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_doc_budgets(tmp_path, report)
    exceeded = [item.message for item in report.warnings if item.code == "budget.exceeded"]
    assert exceeded == ["AGENTS.md: 11 characters exceeds ceiling 10"]
    assert report.errors == ()
    assert [row.path for row in report.budget_rows] == ["AGENTS.md"]


def test_budget_listing_reports_over_without_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    (tmp_path / "scripts/spec").mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("x" * 11, encoding="utf-8")
    (tmp_path / "scripts/spec/doc-budgets.yaml").write_text(
        "schema_version: 2\n"
        "unit: unicode_characters\n"
        "files:\n"
        "  AGENTS.md: 10\n",
        encoding="utf-8",
    )
    assert module.main(["--root", str(tmp_path), "--list-budgets"]) == 0
    output = capsys.readouterr().out
    assert "BUDGET OVER AGENTS.md: 11/10 standing document" in output
    assert "[WARNING] budget.exceeded" in output


def test_invalid_budget_schema_still_fails(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "scripts/spec").mkdir(parents=True)
    (tmp_path / "scripts/spec/doc-budgets.yaml").write_text(
        "schema_version: 99\nunit: unicode_characters\nfiles: {}\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_doc_budgets(tmp_path, report)
    assert "budget.schema" in {item.code for item in report.errors}
    assert module.main(["--root", str(tmp_path), "--list-budgets"]) == 1


def test_historical_bindings_are_not_standing_or_persistence_paths(tmp_path: Path) -> None:
    module = _module()
    entry = _write_binding(tmp_path, "historical")
    registry = {"schema_version": 3, "experiments": {"historical": entry}}
    scientific_paths = module._experiment_binding_paths(entry)
    maintained = {
        path.relative_to(tmp_path).as_posix()
        for path in module.maintained_markdown_paths(tmp_path, registry)
    }
    persisted = module.persistence_paths(tmp_path, registry)
    assert entry["protocol_sources"][0] not in maintained
    assert scientific_paths.isdisjoint(persisted)


def test_skill_validation_auto_discovers_skills_and_allows_resources(tmp_path: Path) -> None:
    module = _module()
    _write_skill(tmp_path, "first-skill")
    _write_skill(tmp_path, "second-skill", add_reference=True)
    report = module.VerificationReport()
    module.validate_skills(tmp_path, report)
    assert report.errors == ()


def test_skill_validation_allows_missing_openai_metadata(tmp_path: Path) -> None:
    module = _module()
    _write_skill(tmp_path, "missing-interface", include_openai=False)
    report = module.VerificationReport()
    module.validate_skills(tmp_path, report)
    assert report.errors == ()


def test_skill_resources_are_checked_for_git_persistence(tmp_path: Path) -> None:
    module = _module()
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Spec Test")
    _git(tmp_path, "config", "user.email", "spec-test@example.invalid")
    skill = _write_skill(tmp_path, "resource-skill", add_reference=True)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    resources = [
        skill / "references/example.md",
        skill / "scripts/check.py",
        skill / "assets/template.json",
    ]
    for path in resources[1:]:
        path.parent.mkdir()
        path.write_text("{}\n")
    resources[0].write_text("# Modified reference\n")
    cache = skill / "__pycache__/check.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"cache")
    (tmp_path / ".gitignore").write_text("__pycache__/\n")

    report = module.VerificationReport()
    module.validate_committed(tmp_path, None, report)
    for path, code in zip(resources, ["persistence.modified", "persistence.untracked", "persistence.untracked"]):
        relative = path.relative_to(tmp_path).as_posix()
        assert any(item.code == code and relative in item.message for item in report.errors)
    assert cache.relative_to(tmp_path).as_posix() not in module.persistence_paths(tmp_path, None)

    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "resources")
    resources[1].unlink()
    report = module.VerificationReport()
    module.validate_committed(tmp_path, None, report)
    assert any(
        item.code == "persistence.modified" and "scripts/check.py" in item.message
        for item in report.errors
    )
    _git(tmp_path, "rm", "--cached", resources[1].relative_to(tmp_path).as_posix())
    report = module.VerificationReport()
    module.validate_committed(tmp_path, None, report)
    assert any(
        item.code == "persistence.untracked" and "scripts/check.py" in item.message
        for item in report.errors
    )


def test_skill_reference_links_are_checked(tmp_path: Path) -> None:
    module = _module()
    skill = _write_skill(tmp_path, "resource-skill", add_reference=True)
    (skill / "references/example.md").write_text("[Missing helper](../scripts/missing.py)\n")
    report = module.VerificationReport()
    module.validate_markdown_links(tmp_path, None, report)
    assert any(
        item.code == "markdown.link_missing" and "scripts/missing.py" in item.message
        for item in report.errors
    )


def test_objective_validator_tests_run_in_filtered_public_tree(tmp_path: Path) -> None:
    module = _module()
    for relative in (
        "scripts/spec/validate_objective_feasibility.py",
        "tests/test_objective_feasibility_contract.py",
    ):
        assert relative in module.persistence_paths(ROOT, None)
        assert module._publication_path_is_included(relative)
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    assert not (tmp_path / ".agents").exists()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_objective_feasibility_contract.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_skill_validation_allows_omitted_policy(tmp_path: Path) -> None:
    module = _module()
    _write_skill(tmp_path, "no-policy", include_policy=False)
    report = module.VerificationReport()
    module.validate_skills(tmp_path, report)
    assert report.errors == ()


def test_skill_invocation_policy_must_be_boolean(tmp_path: Path) -> None:
    module = _module()
    _write_skill(tmp_path, "invalid-policy", policy_value="automatic")
    report = module.VerificationReport()
    module.validate_skills(tmp_path, report)
    assert "skill.policy" in {item.code for item in report.errors}


def test_agent_note_lifecycle_formats_are_valid(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    _write_note(notes / "proposed/2026-08-19-proposal.md", lifecycle="proposed")
    _write_note(notes / "rejected/2026-08-19-rejected.md", lifecycle="rejected")
    _write_note(
        notes / "implemented/2026-08-20-verified.md",
        lifecycle="implemented",
        include_verification=True,
    )
    archived = notes / "archived/2026-08-17-archived.md"
    _write_note(archived, lifecycle="archived")
    manifest = {
        "schema_version": 1,
        "algorithm": "sha256",
        "files": {
            ".agents/notes/archived/2026-08-17-archived.md": hashlib.sha256(
                archived.read_bytes()
            ).hexdigest()
        },
    }
    (notes / "archived/manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert report.errors == ()


def test_agent_notes_allow_zero_implemented_notes(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    (notes / "implemented/2026-08-18-base.md").unlink()
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert report.errors == ()


def test_empty_archive_needs_no_manifest(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    assert not (notes / "archived/manifest.json").exists()
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert report.errors == ()


@pytest.mark.parametrize(
    "heading",
    [
        "## Proposal",
        "## Plan",
        "## Migration plan",
        "## Acceptance criteria",
        "## pLaN",
    ],
)
def test_implemented_note_rejects_proposal_era_heading(
    tmp_path: Path,
    heading: str,
) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    note = notes / "implemented/2026-08-18-base.md"
    note.write_text(
        note.read_text(encoding="utf-8") + f"\n{heading}\n\nFuture work.\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert "note.implemented_present" in {item.code for item in report.errors}


def test_proposal_headings_remain_valid_for_proposed_and_rejected_notes(
    tmp_path: Path,
) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    _write_note(notes / "proposed/2026-08-19-proposal.md", lifecycle="proposed")
    _write_note(notes / "rejected/2026-08-19-rejected.md", lifecycle="rejected")
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert report.errors == ()


def test_implemented_note_ignores_headings_inside_fenced_example(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    note = notes / "implemented/2026-08-18-base.md"
    note.write_text(
        note.read_text(encoding="utf-8")
        + "\n```markdown\n## Plan\n\nExample only.\n```\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert report.errors == ()


def test_archived_note_is_not_rechecked_for_proposal_era_headings(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    archived = notes / "archived/2026-08-17-archived.md"
    _write_note(archived, lifecycle="archived")
    archived.write_text(
        archived.read_text(encoding="utf-8") + "\n## Plan\n\nFrozen history.\n",
        encoding="utf-8",
    )
    relative = ".agents/notes/archived/2026-08-17-archived.md"
    (notes / "archived/manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "sha256",
                "files": {relative: hashlib.sha256(archived.read_bytes()).hexdigest()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert report.errors == ()


def test_archived_note_body_is_not_retrofitted_to_active_format(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    archived = notes / "archived/2026-08-17-legacy.md"
    archived.write_text(
        "# Agent Note: Legacy snapshot\n\n"
        "Status: implemented\n"
        "Archived: 2026-08-20\n\n"
        "## Historical rationale\n\n"
        "This frozen body predates the active template.\n",
        encoding="utf-8",
    )
    relative = ".agents/notes/archived/2026-08-17-legacy.md"
    (notes / "archived/manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "sha256",
                "files": {relative: hashlib.sha256(archived.read_bytes()).hexdigest()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert report.errors == ()


@pytest.mark.parametrize("heading", ["## Testing", "## Verification"])
def test_implemented_note_allows_present_state_evidence_sections(
    tmp_path: Path,
    heading: str,
) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    note = notes / "implemented/2026-08-18-base.md"
    note.write_text(
        note.read_text(encoding="utf-8") + f"\n{heading}\n\nThe check passes.\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert report.errors == ()


def test_agent_note_rejects_legacy_frontmatter(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    legacy = notes / "proposed/2026-08-20-legacy.md"
    legacy.parent.mkdir()
    legacy.write_text(
        "---\nstatus: proposed\ndate: 2026-08-20\n---\n\n"
        "# Agent Note: Legacy\n\nStatus: proposed\n\n## Problem\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert "note.frontmatter" in {item.code for item in report.errors}


def test_rejected_note_requires_status_reason(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    _write_note(
        notes / "rejected/2026-08-20-rejected.md",
        lifecycle="rejected",
        rejection_reason="",
    )
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert "note.status" in {item.code for item in report.errors}


def test_archived_note_requires_iso_date(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    archived = notes / "archived/2026-08-17-archived.md"
    _write_note(archived, lifecycle="archived", archived_date="2026-99-99")
    relative = ".agents/notes/archived/2026-08-17-archived.md"
    (notes / "archived/manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "sha256",
                "files": {relative: hashlib.sha256(archived.read_bytes()).hexdigest()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert "note.archived" in {item.code for item in report.errors}


def test_agent_note_filename_date_must_be_valid(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    _write_note(notes / "implemented/2026-99-99-invalid.md", lifecycle="implemented")
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert "note.filename" in {item.code for item in report.errors}


def test_agent_note_relative_link_must_exist(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    note = notes / "implemented/2026-08-18-base.md"
    note.write_text(
        note.read_text(encoding="utf-8")
        + "\n[Superseded decision](2026-08-01-missing.md)\n",
        encoding="utf-8",
    )
    module.maintained_markdown_paths = lambda root, registry: [note]
    report = module.VerificationReport()
    module.validate_markdown_links(tmp_path, None, report)
    assert "markdown.link_missing" in {item.code for item in report.errors}


@pytest.mark.parametrize("mode", ["mismatch", "deleted", "unsealed"])
def test_archive_fails_closed_for_drift(tmp_path: Path, mode: str) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    archived = notes / "archived/2026-08-17-archived.md"
    _write_note(archived, lifecycle="archived")
    relative = ".agents/notes/archived/2026-08-17-archived.md"
    files: dict[str, str] = {}
    if mode == "mismatch":
        files[relative] = "0" * 64
    elif mode == "deleted":
        archived.unlink()
        files[relative] = "0" * 64
    manifest = {"schema_version": 1, "algorithm": "sha256", "files": files}
    (notes / "archived/manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    codes = {item.code for item in report.errors}
    expected = {
        "mismatch": "archive.hash",
        "deleted": "archive.deleted",
        "unsealed": "archive.unsealed",
    }[mode]
    assert expected in codes


def test_archived_note_requires_manifest_before_default_check(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    _write_note(notes / "archived/2026-08-17-archived.md", lifecycle="archived")
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert "archive.manifest" in {item.code for item in report.errors}


def test_seal_archive_appends_new_hash(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    archived = notes / "archived/2026-08-17-archived.md"
    _write_note(archived, lifecycle="archived")
    report = module.seal_archive(tmp_path)
    assert report.errors == ()
    manifest = json.loads((notes / "archived/manifest.json").read_text(encoding="utf-8"))
    relative = ".agents/notes/archived/2026-08-17-archived.md"
    assert manifest["files"][relative] == hashlib.sha256(archived.read_bytes()).hexdigest()


def test_seal_empty_archive_does_not_create_manifest(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    report = module.seal_archive(tmp_path)
    assert report.errors == ()
    assert not (notes / "archived/manifest.json").exists()


def test_archive_manifest_must_extend_head(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    archived = notes / "archived/2026-08-17-archived.md"
    _write_note(archived, lifecycle="archived")
    relative = ".agents/notes/archived/2026-08-17-archived.md"
    digest = hashlib.sha256(archived.read_bytes()).hexdigest()
    manifest_path = notes / "archived/manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"schema_version": 1, "algorithm": "sha256", "files": {relative: digest}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Spec Test")
    _git(tmp_path, "config", "user.email", "spec-test@example.invalid")
    _git(tmp_path, "add", ".agents")
    _git(tmp_path, "commit", "-q", "-m", "seal")
    manifest_path.write_text(
        json.dumps(
            {"schema_version": 1, "algorithm": "sha256", "files": {relative: "0" * 64}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert "archive.monotonic" in {item.code for item in report.errors}


def test_sealed_head_archive_cannot_remove_manifest_and_files(tmp_path: Path) -> None:
    module = _module()
    notes = _setup_notes(tmp_path)
    archived = notes / "archived/2026-08-17-archived.md"
    _write_note(archived, lifecycle="archived")
    relative = ".agents/notes/archived/2026-08-17-archived.md"
    manifest_path = notes / "archived/manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "sha256",
                "files": {relative: hashlib.sha256(archived.read_bytes()).hexdigest()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Spec Test")
    _git(tmp_path, "config", "user.email", "spec-test@example.invalid")
    _git(tmp_path, "add", ".agents")
    _git(tmp_path, "commit", "-q", "-m", "seal")
    archived.unlink()
    manifest_path.unlink()
    report = module.VerificationReport()
    module.validate_agent_notes(tmp_path, report)
    assert "archive.manifest" in {item.code for item in report.errors}


def test_committed_gate_rejects_untracked_and_modified_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Spec Test")
    _git(tmp_path, "config", "user.email", "spec-test@example.invalid")
    (tmp_path / "tracked.txt").write_text("v1\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "tracked.txt").write_text("v2\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("new\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "persistence_paths",
        lambda root, registry: {"tracked.txt", "untracked.txt"},
    )
    report = module.VerificationReport()
    module.validate_committed(tmp_path, None, report)
    codes = {item.code for item in report.errors}
    assert "persistence.modified" in codes
    assert "persistence.untracked" in codes


def test_committed_gate_handles_unicode_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Spec Test")
    _git(tmp_path, "config", "user.email", "spec-test@example.invalid")
    path = tmp_path / "docs/协议.md"
    path.parent.mkdir()
    path.write_text("# Protocol\n", encoding="utf-8")
    _git(tmp_path, "add", "docs/协议.md")
    _git(tmp_path, "commit", "-q", "-m", "unicode path")
    monkeypatch.setattr(
        module,
        "persistence_paths",
        lambda root, registry: {"docs/协议.md"},
    )
    report = module.VerificationReport()
    module.validate_committed(tmp_path, None, report)
    assert report.errors == ()


def test_strict_history_uses_synthetic_missing_source(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/history.yaml").write_text(
        "sources:\n  protocol: docs/missing.md\n",
        encoding="utf-8",
    )
    report = module.VerificationReport()
    module.audit_history(tmp_path, report)
    messages = [item.message for item in report.errors if item.code == "history.missing_source"]
    assert messages == ["configs/history.yaml references missing repository source docs/missing.md"]


def test_generated_index_interface_is_removed() -> None:
    module = _module()
    assert not hasattr(module, "build_index_content")
    assert not hasattr(module, "GENERATED_INDEX_PATH")
    assert not (ROOT / "docs/generated/spec-index.md").exists()
