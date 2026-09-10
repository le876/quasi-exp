#!/usr/bin/env python3
"""Validate the quasi-exp repository harness.

The default mode proves governance structure only. Scientific contract review
and Git persistence are separate claims.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

import yaml


REGISTRY_PATH = "spec/registry.yaml"
RELEASE_MAP_PATH = "spec/release-map.yaml"
CURRENT_STATE_PATH = "docs/current-state.md"
BUDGET_PATH = "scripts/spec/doc-budgets.yaml"
ARCHIVE_MANIFEST_PATH = ".agents/notes/archived/manifest.json"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NOTE_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]*\.md$")
NOTE_TITLE_RE = re.compile(r"^# Agent Note: \S.*$")
REJECTED_STATUS_RE = re.compile(r"^Status: rejected — \S.*$")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
REPOSITORY_SOURCE_PREFIXES = ("docs/", "configs/", "scripts/", "src/", "tests/")
TEST_TIERS = {"self_contained", "artifact_bound"}
REGISTRY_KEYS = {"schema_version", "experiments"}
EXPERIMENT_KEYS = {
    "role",
    "scientific_source_fixed_point",
    "protocol_sources",
    "config",
    "runner",
    "launcher",
    "tests",
    "upstream_experiments",
}
CURRENT_STATE_KEYS = {
    "schema_version",
    "observed_at",
    "current_experiment",
    "current_attempt",
    "run_root",
    "runtime_task",
    "runtime_state_locator",
    "runtime_log_locator",
    "summary_artifact_locator",
    "current_consultation_locator",
}
FORBIDDEN_STATE_KEYS = {
    "active_experiment",
    "scientific_source_fixed_point",
    "runtime_status",
    "scientific_gate_status",
    "conditional_downstream_status",
    "gate_pass",
    "scientific_gate_pass",
    "authorization",
    "deployment_authorized",
    "repaired_5k_authorized",
    "formal_ready",
    "result_source",
}
RELEASE_MAP_KEYS = {
    "schema_version",
    "public_repository",
    "current_public_release_id",
    "verified_at",
    "mappings",
    "unpublished_scientific_fixed_points",
}
RELEASE_MAPPING_KEYS = {
    "id",
    "status",
    "scientific_source_sha",
    "public_release_sha",
    "evidence",
}
RELEASE_EVIDENCE_KEYS = {"type", "manifest", "commit_url"}
PUBLICATION_MANIFEST_KEYS = {
    "schema_version",
    "source_sha",
    "public_sha",
    "filter_policy",
    "inventory_algorithm",
    "file_count",
    "inventory_sha256",
}
PUBLICATION_EVIDENCE_TYPE = "deterministic_publication_manifest"
PUBLICATION_POLICY = "scientific_public_snapshot_v1"
PUBLICATION_INVENTORY_ALGORITHM = "path-mode-type-content-sha256-v1"
PUBLICATION_MANIFEST_PREFIX = "spec/publication-manifests/"
PUBLICATION_INCLUDE_EXACT = frozenset({".gitignore", "pyproject.toml", "requirements.txt"})
PUBLICATION_INCLUDE_PREFIXES = (
    "configs/",
    "data/robot/",
    "env_specs/",
    "scripts/",
    "src/",
    "tests/",
)
BUDGET_KEYS = {
    "schema_version",
    "unit",
    "files",
}
STANDING_MARKDOWN = {
    "AGENTS.md",
    "README.md",
    "ARCHITECTURE.md",
    "CONTEXT.md",
    "docs/AGENTS.md",
    "docs/README.md",
    "docs/current-state.md",
    "docs/scientific-objective.md",
    "docs/testing.md",
    "docs/protocols/README.md",
    ".agents/notes/README.md",
}
IMPLEMENTED_HEADINGS = (
    "## Problem",
    "## Decision",
    "## Alternatives considered",
    "## Consequences",
)
PROPOSED_HEADINGS = (
    "## Problem",
    "## Proposal",
    "## Alternatives considered",
    "## Acceptance criteria",
    "## Risks",
)
REJECTED_HEADINGS = (
    "## Problem",
    "## Proposal",
    "## Alternatives considered",
)
BANNED_IMPLEMENTED_HEADING_RE = re.compile(
    r"^## (?:Proposal\b|Plan\b|Migration plan\b|Acceptance criteria\b)",
    flags=re.IGNORECASE,
)
MARKDOWN_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class BudgetRow:
    path: str
    used: int
    ceiling: int
    kind: str


@dataclass(frozen=True)
class PublicationInventory:
    records: Mapping[str, tuple[str, str, str]]
    file_count: int
    digest: str


@dataclass
class VerificationReport:
    findings: list[Finding] = field(default_factory=list)
    budget_rows: list[BudgetRow] = field(default_factory=list)

    def error(self, code: str, message: str) -> None:
        self.findings.append(Finding("error", code, message))

    def warning(self, code: str, message: str) -> None:
        self.findings.append(Finding("warning", code, message))

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.severity == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.severity == "warning")


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_yaml(
    path: Path,
    report: VerificationReport,
    label: str,
) -> Mapping[str, Any] | None:
    if not path.is_file():
        report.error("path.missing", f"{label}: missing file {_relative(path.parent, path)}")
        return None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        report.error("yaml.invalid", f"{label}: cannot parse {path}: {exc}")
        return None
    if not isinstance(value, Mapping):
        report.error("yaml.mapping", f"{label}: expected a YAML mapping in {path}")
        return None
    return value


def _read_yaml_text(
    text: str,
    report: VerificationReport,
    label: str,
) -> Mapping[str, Any] | None:
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        report.error("yaml.invalid", f"{label}: cannot parse YAML: {exc}")
        return None
    if not isinstance(value, Mapping):
        report.error("yaml.mapping", f"{label}: expected a YAML mapping")
        return None
    return value


def _read_frontmatter(
    path: Path,
    report: VerificationReport,
    label: str,
) -> tuple[Mapping[str, Any] | None, str]:
    if not path.is_file():
        report.error("path.missing", f"{label}: missing file {path}")
        return None, ""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        report.error("markdown.read", f"{label}: cannot read {path}: {exc}")
        return None, ""
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        report.error("frontmatter.missing", f"{label}: must start with YAML frontmatter")
        return None, text
    closing = next((index for index in range(1, len(lines)) if lines[index] == "---"), None)
    if closing is None:
        report.error("frontmatter.unclosed", f"{label}: missing closing frontmatter marker")
        return None, text
    try:
        value = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        report.error("frontmatter.invalid", f"{label}: {exc}")
        return None, text
    if not isinstance(value, Mapping):
        report.error("frontmatter.mapping", f"{label}: frontmatter must be a mapping")
        return None, text
    return value, text


def _git(root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_bytes(
    root: Path,
    args: Sequence[str],
    *,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            input=input_bytes,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def resolve_project_path(
    root: Path,
    value: Any,
    report: VerificationReport,
    field_name: str,
    *,
    require_exists: bool,
    require_file: bool = True,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        report.error("path.type", f"{field_name}: expected a non-empty repository-relative path")
        return None
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        report.error("path.escape", f"{field_name}: unsafe repository path {value!r}")
        return None
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        report.error("path.escape", f"{field_name}: path escapes repository: {value!r}")
        return None
    if require_exists and not target.exists():
        report.error("path.missing", f"{field_name}: missing {value}")
    elif require_exists and require_file and not target.is_file():
        report.error("path.not_file", f"{field_name}: expected a file at {value}")
    return target


def _nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _nested_strings(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            yield from _nested_strings(nested)


def _experiment_binding_paths(entry: Mapping[str, Any]) -> set[str]:
    paths: set[str] = set()
    protocols = entry.get("protocol_sources")
    if isinstance(protocols, list):
        paths.update(value for value in protocols if isinstance(value, str))
    for key in ("config", "runner", "launcher"):
        value = entry.get(key)
        if isinstance(value, str):
            paths.add(value)
    tests = entry.get("tests")
    if isinstance(tests, Mapping):
        for tier in TEST_TIERS:
            values = tests.get(tier)
            if isinstance(values, list):
                paths.update(value for value in values if isinstance(value, str))
    return paths


def validate_experiment(
    root: Path,
    experiment_id: str,
    entry: Any,
    report: VerificationReport,
) -> None:
    label = f"registry.experiments.{experiment_id}"
    if not isinstance(entry, Mapping):
        report.error("registry.experiment", f"{label}: expected a mapping")
        return
    unknown = sorted(set(entry) - EXPERIMENT_KEYS)
    missing = sorted(EXPERIMENT_KEYS - set(entry))
    if unknown:
        report.error("registry.schema", f"{label}: unsupported fields {unknown}")
    if missing:
        report.error("registry.schema", f"{label}: missing required fields {missing}")
    if not isinstance(entry.get("role"), str) or not entry.get("role", "").strip():
        report.error("registry.role", f"{label}.role: expected a non-empty string")
    source_sha = entry.get("scientific_source_fixed_point")
    if not isinstance(source_sha, str) or SHA_RE.fullmatch(source_sha) is None:
        report.error("registry.source_sha", f"{label}: invalid scientific source SHA")

    protocols = entry.get("protocol_sources")
    if not isinstance(protocols, list) or not protocols:
        report.error("registry.protocols", f"{label}.protocol_sources: expected a non-empty list")
        protocols = []
    if len({value for value in protocols if isinstance(value, str)}) != len(protocols):
        report.error("registry.protocols", f"{label}.protocol_sources must be unique strings")
    for index, value in enumerate(protocols):
        resolve_project_path(
            root,
            value,
            report,
            f"{label}.protocol_sources[{index}]",
            require_exists=False,
        )

    config_value = entry.get("config")
    if config_value is not None:
        resolve_project_path(
            root,
            config_value,
            report,
            f"{label}.config",
            require_exists=False,
        )
    resolve_project_path(
        root,
        entry.get("runner"),
        report,
        f"{label}.runner",
        require_exists=False,
    )
    launcher = entry.get("launcher")
    if launcher is not None:
        resolve_project_path(
            root,
            launcher,
            report,
            f"{label}.launcher",
            require_exists=False,
        )

    tests = entry.get("tests")
    if not isinstance(tests, Mapping) or set(tests) != TEST_TIERS:
        report.error("registry.tests", f"{label}.tests must contain exactly {sorted(TEST_TIERS)}")
    else:
        for tier in sorted(TEST_TIERS):
            values = tests.get(tier)
            if not isinstance(values, list):
                report.error("registry.tests", f"{label}.tests.{tier}: expected a list")
                continue
            if len({value for value in values if isinstance(value, str)}) != len(values):
                report.error("registry.tests", f"{label}.tests.{tier}: paths must be unique strings")
            for index, value in enumerate(values):
                resolve_project_path(
                    root,
                    value,
                    report,
                    f"{label}.tests.{tier}[{index}]",
                    require_exists=False,
                )

    upstream = entry.get("upstream_experiments")
    if not isinstance(upstream, list):
        report.error("registry.upstream", f"{label}.upstream_experiments: expected a list")
    elif len({value for value in upstream if isinstance(value, str)}) != len(upstream):
        report.error(
            "registry.upstream",
            f"{label}.upstream_experiments must contain unique strings",
        )


def validate_registry(root: Path, report: VerificationReport) -> Mapping[str, Any] | None:
    registry = _read_yaml(root / REGISTRY_PATH, report, "registry")
    if registry is None:
        return None
    if set(registry) != REGISTRY_KEYS:
        report.error("registry.schema", f"registry fields must be exactly {sorted(REGISTRY_KEYS)}")
    if registry.get("schema_version") != 3:
        report.error("registry.schema", "registry.schema_version must be 3")
    experiments = registry.get("experiments")
    if not isinstance(experiments, Mapping) or not experiments:
        report.error("registry.experiments", "registry.experiments must be a non-empty mapping")
        return registry

    for experiment_id, entry in experiments.items():
        if not isinstance(experiment_id, str) or not experiment_id:
            report.error("registry.experiment_key", f"invalid experiment key {experiment_id!r}")
            continue
        validate_experiment(root, experiment_id, entry, report)

    graph: dict[str, tuple[str, ...]] = {}
    for experiment_id, entry in experiments.items():
        if not isinstance(experiment_id, str) or not isinstance(entry, Mapping):
            continue
        upstream = entry.get("upstream_experiments")
        if not isinstance(upstream, list) or not all(isinstance(value, str) for value in upstream):
            continue
        graph[experiment_id] = tuple(upstream)
        for upstream_id in upstream:
            if upstream_id not in experiments:
                report.error(
                    "registry.upstream",
                    f"registry.experiments.{experiment_id}: unknown upstream {upstream_id!r}",
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(experiment_id: str, trail: tuple[str, ...]) -> None:
        if experiment_id in visited:
            return
        if experiment_id in visiting:
            cycle_start = trail.index(experiment_id) if experiment_id in trail else 0
            cycle = (*trail[cycle_start:], experiment_id)
            report.error("registry.upstream_cycle", "upstream cycle: " + " -> ".join(cycle))
            return
        visiting.add(experiment_id)
        for upstream_id in graph.get(experiment_id, ()):
            if upstream_id in graph:
                visit(upstream_id, (*trail, experiment_id))
        visiting.remove(experiment_id)
        visited.add(experiment_id)

    for experiment_id in graph:
        visit(experiment_id, ())
    return registry


def validate_fixed_point_bindings(
    root: Path,
    registry: Mapping[str, Any] | None,
    report: VerificationReport,
) -> None:
    if not isinstance(registry, Mapping):
        return
    probe = _git(root, ["rev-parse", "--is-inside-work-tree"])
    if probe is None or probe.returncode != 0 or probe.stdout.strip() != "true":
        report.error("git.unavailable", "repository Git fixed-point checks are unavailable")
        return
    experiments = registry.get("experiments")
    if not isinstance(experiments, Mapping):
        return
    commit_validity: dict[str, bool] = {}
    path_validity: dict[tuple[str, str], bool] = {}

    def source_path_exists(
        source_sha: str,
        relative: str,
        *,
        label: str,
        code: str,
    ) -> bool:
        key = (source_sha, relative)
        if key not in path_validity:
            result = _git(root, ["cat-file", "-e", source_sha + ":" + relative])
            if result is None:
                report.error("git.unavailable", f"cannot inspect {label} at {source_sha}")
                path_validity[key] = False
            else:
                path_validity[key] = result.returncode == 0
        if not path_validity[key]:
            report.error(code, f"{label}: {relative} does not exist at scientific source {source_sha}")
        return path_validity[key]

    for experiment_id, entry in experiments.items():
        if not isinstance(entry, Mapping):
            continue
        source_sha = entry.get("scientific_source_fixed_point")
        if not isinstance(source_sha, str) or SHA_RE.fullmatch(source_sha) is None:
            continue
        if source_sha not in commit_validity:
            result = _git(root, ["cat-file", "-e", source_sha + "^{commit}"])
            commit_validity[source_sha] = bool(result and result.returncode == 0)
            if not commit_validity[source_sha]:
                report.error(
                    "git.source_missing",
                    f"scientific source fixed point is not a local Git commit: {source_sha}",
                )
        if not commit_validity[source_sha]:
            continue
        label = f"registry.experiments.{experiment_id}"
        for relative in sorted(_experiment_binding_paths(entry)):
            source_path_exists(
                source_sha,
                relative,
                label=experiment_id,
                code="git.binding_missing",
            )

        config_path = entry.get("config")
        if not isinstance(config_path, str) or not path_validity.get((source_sha, config_path), False):
            continue
        shown = _git(root, ["show", source_sha + ":" + config_path])
        if shown is None or shown.returncode != 0:
            report.error("git.unavailable", f"{label}: cannot read source-bound config")
            continue
        config = _read_yaml_text(shown.stdout, report, f"{label}.config@{source_sha}")
        if config is None:
            continue
        if "experiment_id" in config and config.get("experiment_id") != experiment_id:
            report.error(
                "registry.experiment_id",
                f"{label}: source-bound config experiment_id {config.get('experiment_id')!r} does not match key",
            )

        robot_config = config.get("robot_config")
        if robot_config is not None:
            robot_path = resolve_project_path(
                root,
                robot_config,
                report,
                f"{label}.config.robot_config",
                require_exists=False,
            )
            if robot_path is not None and isinstance(robot_config, str):
                source_path_exists(
                    source_sha,
                    robot_config,
                    label=f"{label}.config.robot_config",
                    code="git.source_inventory_missing",
                )

        registered_protocols = {
            value for value in entry.get("protocol_sources", []) if isinstance(value, str)
        }
        config_protocols: set[str] = set()
        source_inventory = config.get("sources")
        if isinstance(source_inventory, Mapping):
            for source in _nested_strings(source_inventory):
                if source.startswith(REPOSITORY_SOURCE_PREFIXES):
                    source_path = resolve_project_path(
                        root,
                        source,
                        report,
                        f"{label}.config.sources",
                        require_exists=False,
                    )
                    if source_path is not None:
                        source_path_exists(
                            source_sha,
                            source,
                            label=f"{label}.config.sources",
                            code="git.source_inventory_missing",
                        )
                if source.startswith("docs/") and source.endswith(".md"):
                    config_protocols.add(source)

        absent_from_registry = sorted(config_protocols - registered_protocols)
        if absent_from_registry:
            report.error(
                "registry.protocol_binding",
                f"{label}: source-bound config protocols absent from registry: {absent_from_registry}",
            )
        absent_from_inventory = sorted(registered_protocols - config_protocols)
        if absent_from_inventory:
            report.warning(
                "scientific.protocol_inventory",
                f"{label}: registry protocols absent from source-bound config inventory: {absent_from_inventory}",
            )


def _publication_path_is_included(path: str) -> bool:
    return (
        path in PUBLICATION_INCLUDE_EXACT or path.startswith(PUBLICATION_INCLUDE_PREFIXES)
    ) and not path.endswith(".md")


def _git_commit_available(
    root: Path,
    commit: str,
    *,
    kind: str,
    label: str,
    report: VerificationReport,
) -> bool:
    result = _git(root, ["cat-file", "-e", f"{commit}^{{commit}}"])
    if result is not None and result.returncode == 0:
        return True
    if kind == "public":
        hint = "; fetch the ref containing it explicitly (for this repository: git fetch origin main)"
    else:
        hint = "; fetch the scientific source history explicitly"
    report.error(
        "release.commit_missing",
        f"{label}: {kind} commit {commit} is unavailable in the local Git object database{hint}",
    )
    return False


def _git_tree_entries(
    root: Path,
    commit: str,
    *,
    label: str,
    report: VerificationReport,
) -> dict[str, tuple[str, str, str]] | None:
    result = _git_bytes(root, ["ls-tree", "-rz", "--full-tree", commit])
    if result is None or result.returncode != 0:
        report.error("release.tree", f"{label}: cannot read Git tree for {commit}")
        return None
    entries: dict[str, tuple[str, str, str]] = {}
    for raw_record in result.stdout.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (UnicodeError, ValueError):
            report.error("release.tree", f"{label}: malformed or non-UTF-8 Git tree entry")
            return None
        if path in entries:
            report.error("release.tree", f"{label}: duplicate Git tree path {path!r}")
            return None
        entries[path] = (mode, object_type, object_id)
    return entries


def _git_blob_sha256s(
    root: Path,
    object_ids: Iterable[str],
    *,
    label: str,
    report: VerificationReport,
) -> dict[str, str] | None:
    ordered = sorted(set(object_ids))
    query = b"".join(object_id.encode("ascii") + b"\n" for object_id in ordered)
    result = _git_bytes(root, ["cat-file", "--batch"], input_bytes=query)
    if result is None or result.returncode != 0:
        report.error("release.content", f"{label}: cannot read publication blobs")
        return None

    hashes: dict[str, str] = {}
    cursor = 0
    output = result.stdout
    for requested_id in ordered:
        header_end = output.find(b"\n", cursor)
        if header_end < 0:
            report.error("release.content", f"{label}: truncated git cat-file response")
            return None
        header = output[cursor:header_end].split()
        if len(header) != 3:
            report.error("release.content", f"{label}: invalid git cat-file response")
            return None
        actual_id, object_type, raw_size = header
        try:
            size = int(raw_size)
        except ValueError:
            report.error("release.content", f"{label}: invalid blob size from git cat-file")
            return None
        if object_type != b"blob":
            report.error(
                "release.entry_type",
                f"{label}: object {requested_id} has unsupported type {object_type.decode('ascii', 'replace')}",
            )
            return None
        content_start = header_end + 1
        content_end = content_start + size
        if content_end >= len(output) or output[content_end : content_end + 1] != b"\n":
            report.error("release.content", f"{label}: truncated blob content from git cat-file")
            return None
        hashes[actual_id.decode("ascii")] = hashlib.sha256(
            output[content_start:content_end]
        ).hexdigest()
        cursor = content_end + 1
    if cursor != len(output):
        report.error("release.content", f"{label}: unexpected trailing git cat-file output")
        return None
    return hashes


def _publication_inventory(
    root: Path,
    entries: Mapping[str, tuple[str, str, str]],
    *,
    label: str,
    report: VerificationReport,
) -> PublicationInventory | None:
    non_blobs = sorted(
        path
        for path, (_mode, object_type, _oid) in entries.items()
        if object_type != "blob"
    )
    if non_blobs:
        report.error(
            "release.entry_type",
            f"{label}: publication inventory contains non-blob entries {non_blobs[:5]}",
        )
        return None
    content_hashes = _git_blob_sha256s(
        root,
        (object_id for _mode, _type, object_id in entries.values()),
        label=label,
        report=report,
    )
    if content_hashes is None:
        return None

    records: dict[str, tuple[str, str, str]] = {}
    payload = bytearray()
    for path in sorted(entries, key=lambda value: value.encode("utf-8")):
        mode, object_type, object_id = entries[path]
        content_sha256 = content_hashes.get(object_id)
        if content_sha256 is None:
            report.error("release.content", f"{label}: missing content hash for {path}")
            return None
        records[path] = (mode, object_type, content_sha256)
        payload.extend(path.encode("utf-8"))
        payload.extend(b"\0")
        payload.extend(mode.encode("ascii"))
        payload.extend(b"\0")
        payload.extend(object_type.encode("ascii"))
        payload.extend(b"\0")
        payload.extend(content_sha256.encode("ascii"))
        payload.extend(b"\n")
    return PublicationInventory(
        records=records,
        file_count=len(records),
        digest=hashlib.sha256(payload).hexdigest(),
    )


def _validate_publication_manifest(
    root: Path,
    *,
    label: str,
    source_sha: str,
    public_sha: str,
    manifest_value: Any,
    report: VerificationReport,
) -> None:
    manifest_path = resolve_project_path(
        root,
        manifest_value,
        report,
        f"{label}.evidence.manifest",
        require_exists=True,
    )
    if manifest_path is None:
        return
    relative_manifest = _relative(root, manifest_path)
    if not relative_manifest.startswith(
        PUBLICATION_MANIFEST_PREFIX
    ) or not relative_manifest.endswith(".yaml"):
        report.error(
            "release.manifest",
            f"{label}: publication manifest must be a YAML file under {PUBLICATION_MANIFEST_PREFIX}",
        )
        return
    manifest = _read_yaml(manifest_path, report, f"{label}.publication_manifest")
    if manifest is None:
        return
    if set(manifest) != PUBLICATION_MANIFEST_KEYS:
        report.error(
            "release.manifest",
            f"{label}: publication manifest fields must be exactly {sorted(PUBLICATION_MANIFEST_KEYS)}",
        )
    if manifest.get("schema_version") != 1:
        report.error("release.manifest", f"{label}: publication manifest schema_version must be 1")
    if manifest.get("source_sha") != source_sha or manifest.get("public_sha") != public_sha:
        report.error(
            "release.manifest_binding",
            f"{label}: publication manifest SHA binding differs from release map",
        )
        return
    if manifest.get("filter_policy") != PUBLICATION_POLICY:
        report.error("release.policy", f"{label}: unsupported publication filter policy")
        return
    if manifest.get("inventory_algorithm") != PUBLICATION_INVENTORY_ALGORITHM:
        report.error("release.inventory", f"{label}: unsupported publication inventory algorithm")
        return
    expected_count = manifest.get("file_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 0:
        report.error("release.inventory", f"{label}: file_count must be a non-negative integer")
        expected_count = None
    expected_digest = manifest.get("inventory_sha256")
    if not isinstance(expected_digest, str) or SHA256_RE.fullmatch(expected_digest) is None:
        report.error("release.inventory", f"{label}: inventory_sha256 must be a SHA-256 digest")
        expected_digest = None

    source_available = _git_commit_available(
        root,
        source_sha,
        kind="source",
        label=label,
        report=report,
    )
    public_available = _git_commit_available(
        root,
        public_sha,
        kind="public",
        label=label,
        report=report,
    )
    if not source_available or not public_available:
        return
    source_tree = _git_tree_entries(root, source_sha, label=f"{label}.source", report=report)
    public_tree = _git_tree_entries(root, public_sha, label=f"{label}.public", report=report)
    if source_tree is None or public_tree is None:
        return
    filtered_source = {
        path: entry for path, entry in source_tree.items() if _publication_path_is_included(path)
    }
    source_inventory = _publication_inventory(
        root,
        filtered_source,
        label=f"{label}.source",
        report=report,
    )
    public_inventory = _publication_inventory(
        root,
        public_tree,
        label=f"{label}.public",
        report=report,
    )
    if source_inventory is None or public_inventory is None:
        return

    source_paths = set(source_inventory.records)
    public_paths = set(public_inventory.records)
    missing = sorted(source_paths - public_paths)
    extra = sorted(public_paths - source_paths)
    if missing:
        report.error(
            "release.tree",
            f"{label}: public tree is missing filtered source paths {missing[:5]}",
        )
    if extra:
        report.error(
            "release.tree",
            f"{label}: public tree has paths outside filtered source {extra[:5]}",
        )
    mismatched = sorted(
        path
        for path in source_paths & public_paths
        if source_inventory.records[path] != public_inventory.records[path]
    )
    if mismatched:
        report.error(
            "release.tree",
            f"{label}: public tree mode, type, or content differs at {mismatched[:5]}",
        )
    if expected_count is not None and (
        source_inventory.file_count != expected_count
        or public_inventory.file_count != expected_count
    ):
        report.error(
            "release.inventory",
            f"{label}: manifest file_count {expected_count} does not match source/public inventories "
            f"{source_inventory.file_count}/{public_inventory.file_count}",
        )
    if expected_digest is not None and (
        source_inventory.digest != expected_digest or public_inventory.digest != expected_digest
    ):
        report.error(
            "release.inventory",
            f"{label}: manifest inventory_sha256 does not match source/public inventories",
        )


def validate_release_map(
    root: Path,
    registry: Mapping[str, Any] | None,
    report: VerificationReport,
) -> Mapping[str, Any] | None:
    release_map = _read_yaml(root / RELEASE_MAP_PATH, report, "release-map")
    if release_map is None:
        return None
    if set(release_map) != RELEASE_MAP_KEYS:
        report.error("release.schema", f"release-map fields must be exactly {sorted(RELEASE_MAP_KEYS)}")
    if release_map.get("schema_version") != 3:
        report.error("release.schema", "release-map.schema_version must be 3")
    repository = release_map.get("public_repository")
    if not isinstance(repository, str) or not repository.startswith("https://"):
        report.error("release.repository", "public_repository must be an https URL")
    observed = release_map.get("verified_at")
    if not isinstance(observed, str):
        report.error("release.verified_at", "verified_at must be a quoted ISO date")
    else:
        try:
            date.fromisoformat(observed)
        except ValueError:
            report.error("release.verified_at", f"invalid verified_at date {observed!r}")

    mappings = release_map.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        report.error("release.mappings", "mappings must be a non-empty list")
        mappings = []
    by_id: dict[str, Mapping[str, Any]] = {}
    released_sources: set[str] = set()
    public_releases: set[str] = set()
    for index, item in enumerate(mappings):
        label = f"release-map.mappings[{index}]"
        if not isinstance(item, Mapping):
            report.error("release.mapping", f"{label}: expected a mapping")
            continue
        if set(item) != RELEASE_MAPPING_KEYS:
            report.error("release.mapping", f"{label}: invalid fields")
        mapping_id = item.get("id")
        if not isinstance(mapping_id, str) or not mapping_id:
            report.error("release.id", f"{label}: missing id")
        elif mapping_id in by_id:
            report.error("release.id", f"{label}: duplicate id {mapping_id}")
        else:
            by_id[mapping_id] = item
        source_sha = item.get("scientific_source_sha")
        public_sha = item.get("public_release_sha")
        if not isinstance(source_sha, str) or SHA_RE.fullmatch(source_sha) is None:
            report.error("release.source_sha", f"{label}: invalid scientific source SHA")
        elif source_sha in released_sources:
            report.error("release.source_sha", f"{label}: duplicate scientific source SHA")
        else:
            released_sources.add(source_sha)
        if not isinstance(public_sha, str) or SHA_RE.fullmatch(public_sha) is None:
            report.error("release.public_sha", f"{label}: invalid public release SHA")
        elif public_sha in public_releases:
            report.error("release.public_sha", f"{label}: duplicate public release SHA")
        else:
            public_releases.add(public_sha)
        if source_sha == public_sha:
            report.error("release.identity", f"{label}: source and public SHA must differ")
        if item.get("status") != "verified":
            report.error("release.status", f"{label}: unsupported mapping status")
        evidence = item.get("evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != RELEASE_EVIDENCE_KEYS:
            report.error("release.evidence", f"{label}: invalid evidence mapping")
        else:
            expected_url = (
                repository.rstrip("/") + "/commit/" + public_sha
                if isinstance(repository, str) and isinstance(public_sha, str)
                else None
            )
            if evidence.get("type") != PUBLICATION_EVIDENCE_TYPE:
                report.error("release.evidence", f"{label}: unsupported publication evidence type")
            elif evidence.get("commit_url") != expected_url:
                report.error("release.evidence", f"{label}: commit_url does not identify declared commit")
            elif (
                isinstance(source_sha, str)
                and SHA_RE.fullmatch(source_sha) is not None
                and isinstance(public_sha, str)
                and SHA_RE.fullmatch(public_sha) is not None
            ):
                _validate_publication_manifest(
                    root,
                    label=label,
                    source_sha=source_sha,
                    public_sha=public_sha,
                    manifest_value=evidence.get("manifest"),
                    report=report,
                )

    current_id = release_map.get("current_public_release_id")
    if current_id not in by_id:
        report.error("release.current", f"current_public_release_id {current_id!r} is not mapped")

    unpublished = release_map.get("unpublished_scientific_fixed_points")
    if not isinstance(unpublished, list):
        report.error("release.unpublished", "unpublished_scientific_fixed_points must be a list")
        unpublished = []
    unpublished_sources: set[str] = set()
    for index, source_sha in enumerate(unpublished):
        label = f"release-map.unpublished_scientific_fixed_points[{index}]"
        if not isinstance(source_sha, str) or SHA_RE.fullmatch(source_sha) is None:
            report.error("release.source_sha", f"{label}: invalid scientific source SHA")
            continue
        if source_sha in unpublished_sources:
            report.error("release.source_sha", f"{label}: duplicate source SHA")
        unpublished_sources.add(source_sha)
    overlap = sorted(released_sources & unpublished_sources)
    if overlap:
        report.error(
            "release.overlap",
            f"scientific source SHA cannot be both released and unpublished: {overlap}",
        )

    experiments = registry.get("experiments", {}) if isinstance(registry, Mapping) else {}
    if isinstance(experiments, Mapping):
        known_sources = released_sources | unpublished_sources
        for experiment_id, entry in experiments.items():
            if not isinstance(entry, Mapping):
                continue
            source_sha = entry.get("scientific_source_fixed_point")
            if source_sha not in known_sources:
                report.error(
                    "release.coverage",
                    f"experiment {experiment_id} source {source_sha!r} is not declared",
                )
    return release_map


def validate_current_state(
    root: Path,
    registry: Mapping[str, Any] | None,
    report: VerificationReport,
) -> Mapping[str, Any] | None:
    metadata, _text = _read_frontmatter(root / CURRENT_STATE_PATH, report, "current-state")
    if metadata is None:
        return None
    forbidden = sorted(set(metadata) & FORBIDDEN_STATE_KEYS)
    for field_name in forbidden:
        report.error(
            "state.result_field",
            f"current-state must not contain binding/result field {field_name}",
        )
    if set(metadata) != CURRENT_STATE_KEYS:
        report.error(
            "state.schema",
            f"current-state fields must be exactly {sorted(CURRENT_STATE_KEYS)}",
        )
    if metadata.get("schema_version") != 3:
        report.error("state.schema", "current-state.schema_version must be 3")
    observed = metadata.get("observed_at")
    if not isinstance(observed, str):
        report.error("state.timestamp", "observed_at must be a quoted ISO-8601 string")
    else:
        try:
            datetime.fromisoformat(observed)
        except ValueError:
            report.error("state.timestamp", f"invalid observed_at {observed!r}")

    experiments = registry.get("experiments", {}) if isinstance(registry, Mapping) else {}
    current = metadata.get("current_experiment")
    if not isinstance(current, str) or not current.strip():
        report.error("state.experiment", "current_experiment must be a non-empty string")
    elif not isinstance(experiments, Mapping) or current not in experiments:
        report.error("state.experiment", f"current_experiment {current!r} is not in registry")

    for field_name in ("current_attempt", "runtime_task"):
        value = metadata.get(field_name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            report.error("state.pointer", f"{field_name} must be null or a non-empty string")

    for field_name in (
        "run_root",
        "runtime_state_locator",
        "runtime_log_locator",
        "summary_artifact_locator",
    ):
        value = metadata.get(field_name)
        if value is None:
            continue
        resolve_project_path(
            root,
            value,
            report,
            "current-state." + field_name,
            require_exists=False,
            require_file=False,
        )

    consultation = metadata.get("current_consultation_locator")
    if consultation is not None:
        resolve_project_path(
            root,
            consultation,
            report,
            "current-state.current_consultation_locator",
            require_exists=True,
        )
    summary = metadata.get("summary_artifact_locator")
    run_root = metadata.get("run_root")
    if (
        isinstance(summary, str)
        and isinstance(run_root, str)
        and not summary.startswith(run_root.rstrip("/") + "/")
    ):
        report.error("state.artifact_binding", "summary_artifact_locator must live under run_root")
    return metadata


def validate_doc_budgets(
    root: Path,
    report: VerificationReport,
) -> None:
    budget = _read_yaml(root / BUDGET_PATH, report, "doc budgets")
    if budget is None:
        return
    if set(budget) != BUDGET_KEYS:
        report.error("budget.schema", f"doc budget fields must be exactly {sorted(BUDGET_KEYS)}")
    if budget.get("schema_version") != 2 or budget.get("unit") != "unicode_characters":
        report.error("budget.schema", "doc budgets must use schema 2 and unicode_characters")
    file_budgets = budget.get("files")
    if not isinstance(file_budgets, Mapping):
        report.error("budget.schema", "doc budgets files must be a mapping")
        file_budgets = {}

    rows: list[BudgetRow] = []
    for relative, ceiling in file_budgets.items():
        if not isinstance(relative, str) or not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling <= 0:
            report.error("budget.schema", f"invalid file budget {relative!r}: {ceiling!r}")
            continue
        path = resolve_project_path(
            root,
            relative,
            report,
            "doc budgets.files",
            require_exists=True,
        )
        if path is not None and path.is_file():
            rows.append(
                BudgetRow(relative, len(path.read_text(encoding="utf-8")), ceiling, "standing document")
            )

    report.budget_rows = sorted(rows, key=lambda row: row.path)
    for row in report.budget_rows:
        if row.used > row.ceiling:
            report.warning(
                "budget.exceeded",
                f"{row.path}: {row.used} characters exceeds ceiling {row.ceiling}",
            )


def validate_skills(root: Path, report: VerificationReport) -> None:
    skills_root = root / ".agents/skills"
    if not skills_root.is_dir():
        report.error("skill.missing", "missing .agents/skills directory")
        return
    skill_dirs = sorted(path for path in skills_root.iterdir() if path.is_dir())
    for skill_dir in skill_dirs:
        skill_name = skill_dir.name
        files = {
            path.relative_to(skill_dir).as_posix()
            for path in skill_dir.rglob("*")
            if path.is_file()
        }
        missing_files = {"SKILL.md"} - files
        if missing_files:
            report.error("skill.files", f"{skill_name}: missing required files {sorted(missing_files)}")
        metadata, text = _read_frontmatter(skill_dir / "SKILL.md", report, "skill " + skill_name)
        if metadata is None:
            continue
        if set(metadata) != {"name", "description"}:
            report.error("skill.frontmatter", f"{skill_name}: frontmatter must contain name and description")
        if metadata.get("name") != skill_name:
            report.error("skill.name", f"{skill_name}: frontmatter name does not match folder")
        description = metadata.get("description")
        if not isinstance(description, str) or not description.strip() or "TODO" in description:
            report.error("skill.description", f"{skill_name}: description is incomplete")
        if "TODO" in text:
            report.error("skill.todo", f"{skill_name}: SKILL.md contains TODO")
        openai_path = skill_dir / "agents/openai.yaml"
        if not openai_path.is_file():
            continue
        openai = _read_yaml(openai_path, report, "skill UI " + skill_name)
        if openai is None:
            continue
        unknown_openai = sorted(set(openai) - {"interface", "policy", "dependencies"})
        if unknown_openai:
            report.error("skill.interface", f"{skill_name}: unsupported openai.yaml fields {unknown_openai}")
        interface = openai.get("interface")
        policy = openai.get("policy")
        if interface is not None:
            if not isinstance(interface, Mapping) or set(interface) != {
                "display_name",
                "short_description",
                "default_prompt",
            }:
                report.error("skill.interface", f"{skill_name}: invalid interface fields")
            else:
                for key in ("display_name", "short_description", "default_prompt"):
                    if not isinstance(interface.get(key), str) or not interface.get(key, "").strip():
                        report.error("skill.interface", f"{skill_name}: interface.{key} is required")
                short = interface.get("short_description")
                if isinstance(short, str) and not 25 <= len(short) <= 64:
                    report.error("skill.short_description", f"{skill_name}: short_description must be 25-64 characters")
                prompt = interface.get("default_prompt")
                if isinstance(prompt, str) and "$" + skill_name not in prompt:
                    report.error("skill.default_prompt", f"{skill_name}: default_prompt must mention the skill")
        if policy is not None:
            if not isinstance(policy, Mapping) or set(policy) != {"allow_implicit_invocation"}:
                report.error("skill.policy", f"{skill_name}: invalid policy fields")
            elif not isinstance(policy.get("allow_implicit_invocation"), bool):
                report.error("skill.policy", f"{skill_name}: allow_implicit_invocation must be boolean")


def _archive_manifest(
    root: Path,
    report: VerificationReport,
) -> Mapping[str, Any] | None:
    path = root / ARCHIVE_MANIFEST_PATH
    if not path.is_file():
        report.error("archive.manifest", f"missing {ARCHIVE_MANIFEST_PATH}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.error("archive.manifest", f"invalid archive manifest: {exc}")
        return None
    if not isinstance(value, Mapping):
        report.error("archive.manifest", "archive manifest must be an object")
        return None
    if set(value) != {"schema_version", "algorithm", "files"}:
        report.error("archive.manifest", "archive manifest has unsupported fields")
    if value.get("schema_version") != 1 or value.get("algorithm") != "sha256":
        report.error("archive.manifest", "archive manifest must use schema 1 and sha256")
    if not isinstance(value.get("files"), Mapping):
        report.error("archive.manifest", "archive manifest files must be an object")
        return None
    return value


def _head_archive_files(root: Path, report: VerificationReport) -> Mapping[str, Any] | None:
    result = _git(root, ["show", "HEAD:" + ARCHIVE_MANIFEST_PATH])
    if result is None:
        report.error("git.unavailable", "cannot inspect HEAD archive manifest")
        return None
    if result.returncode != 0:
        return {}
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        report.error("archive.head_manifest", f"HEAD archive manifest is invalid: {exc}")
        return None
    files = value.get("files") if isinstance(value, Mapping) else None
    if not isinstance(files, Mapping):
        report.error("archive.head_manifest", "HEAD archive manifest has invalid files")
        return None
    return files


def validate_archive(
    root: Path,
    report: VerificationReport,
    *,
    allow_unsealed: bool,
) -> None:
    archive_dir = root / ".agents/notes/archived"
    archived = {
        _relative(root, path): path
        for path in sorted(archive_dir.glob("*.md"))
        if path.is_file()
    }
    manifest_path = root / ARCHIVE_MANIFEST_PATH
    head_entries = _head_archive_files(root, report)
    if not manifest_path.is_file():
        if isinstance(head_entries, Mapping) and head_entries:
            report.error("archive.manifest", f"missing {ARCHIVE_MANIFEST_PATH} for sealed HEAD archive")
        elif archived and not allow_unsealed:
            report.error("archive.manifest", f"missing {ARCHIVE_MANIFEST_PATH}; seal the archived Notes")
        return

    manifest = _archive_manifest(root, report)
    if manifest is None:
        return
    entries = manifest.get("files")
    if not isinstance(entries, Mapping):
        return
    entry_paths = set(entries)
    archived_paths = set(archived)
    extra = sorted(entry_paths - archived_paths)
    missing = sorted(archived_paths - entry_paths)
    if extra:
        report.error("archive.deleted", f"sealed archive paths are missing: {extra}")
    if missing and not allow_unsealed:
        report.error("archive.unsealed", f"archived Notes lack seals: {missing}")
    for relative in sorted(entry_paths & archived_paths):
        expected = entries.get(relative)
        if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
            report.error("archive.hash", f"{relative}: invalid manifest SHA-256")
            continue
        actual = hashlib.sha256(archived[relative].read_bytes()).hexdigest()
        if actual != expected:
            report.error("archive.hash", f"{relative}: archived content differs from sealed hash")
    if isinstance(head_entries, Mapping):
        for relative, expected in head_entries.items():
            if entries.get(relative) != expected:
                report.error(
                    "archive.monotonic",
                    f"{relative}: working manifest is not a monotonic extension of HEAD",
                )


def _markdown_h2_headings(lines: Sequence[str]) -> list[str]:
    headings: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in lines:
        fence = MARKDOWN_FENCE_RE.match(line)
        if fence is not None:
            marker, remainder = fence.groups()
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif (
                marker[0] == fence_character
                and len(marker) >= fence_length
                and not remainder.strip()
            ):
                fence_character = None
                fence_length = 0
            continue
        if fence_character is None and re.fullmatch(r"## [^\n]+", line):
            headings.append(line.rstrip())
    return headings


def validate_agent_notes(
    root: Path,
    report: VerificationReport,
    *,
    allow_unsealed_archive: bool = False,
) -> None:
    notes_root = root / ".agents/notes"
    if not (notes_root / "README.md").is_file():
        report.error("note.readme", "missing .agents/notes/README.md")
    allowed_dirs = {"proposed", "implemented", "rejected", "archived"}
    if notes_root.is_dir():
        unexpected_dirs = sorted(
            path.name for path in notes_root.iterdir() if path.is_dir() and path.name not in allowed_dirs
        )
        if unexpected_dirs:
            report.error("note.lifecycle", f"unexpected Agent Note directories: {unexpected_dirs}")
        unexpected_files = sorted(
            path.name for path in notes_root.iterdir() if path.is_file() and path.name != "README.md"
        )
        if unexpected_files:
            report.error("note.lifecycle", f"unexpected files at Agent Note root: {unexpected_files}")

    for lifecycle in sorted(allowed_dirs):
        directory = notes_root / lifecycle
        if not directory.exists():
            continue
        allowed_non_markdown = {"manifest.json"} if lifecycle == "archived" else set()
        non_markdown = sorted(
            path.name
            for path in directory.iterdir()
            if path.is_file() and path.suffix != ".md" and path.name not in allowed_non_markdown
        )
        if non_markdown:
            report.error("note.lifecycle", f"unexpected files in {lifecycle}: {non_markdown}")
        for path in sorted(directory.rglob("*.md")):
            relative = _relative(root, path)
            if path.parent != directory:
                report.error("note.nesting", f"Agent Note must be directly under lifecycle: {relative}")
            if NOTE_NAME_RE.fullmatch(path.name) is None:
                report.error("note.filename", f"invalid Agent Note filename: {relative}")
            else:
                try:
                    date.fromisoformat(path.name[:10])
                except ValueError:
                    report.error("note.filename", f"invalid Agent Note date: {relative}")
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                report.error("markdown.read", f"Agent Note {relative}: {exc}")
                continue
            lines = text.splitlines()
            if lines and lines[0] == "---":
                report.error("note.frontmatter", f"{relative}: Agent Notes do not use YAML frontmatter")
                continue
            if len(lines) < 4 or NOTE_TITLE_RE.fullmatch(lines[0]) is None or lines[1] != "":
                report.error("note.header", f"{relative}: invalid Agent Note title/header block")
                continue

            status_line = lines[2]
            if lifecycle in {"implemented", "archived"}:
                status_valid = status_line == "Status: implemented"
            elif lifecycle == "proposed":
                status_valid = status_line == "Status: proposed"
            else:
                status_valid = REJECTED_STATUS_RE.fullmatch(status_line) is not None
            if not status_valid:
                report.error("note.status", f"{relative}: status does not match lifecycle")

            if lifecycle == "archived":
                if len(lines) < 5 or not lines[3].startswith("Archived: ") or lines[4] != "":
                    report.error("note.archived", f"{relative}: invalid Archived line")
                else:
                    try:
                        date.fromisoformat(lines[3].removeprefix("Archived: "))
                    except ValueError:
                        report.error("note.archived", f"{relative}: Archived must be an ISO date")
                continue
            else:
                if lines[3] != "":
                    report.error("note.header", f"{relative}: status must be followed by a blank line")
                body_lines = lines[4:]

            body_headings = _markdown_h2_headings(body_lines)
            if not body_headings or body_headings[0] != "## Problem":
                report.error("note.sections", f"{relative}: first section must be ## Problem")
            if lifecycle in {"implemented", "archived"}:
                required_headings = IMPLEMENTED_HEADINGS
            elif lifecycle == "proposed":
                required_headings = PROPOSED_HEADINGS
            else:
                required_headings = REJECTED_HEADINGS
            missing = [heading for heading in required_headings if heading not in body_headings]
            if missing:
                report.error("note.sections", f"{relative}: missing sections {missing}")
            if lifecycle == "implemented":
                for heading in body_headings:
                    if BANNED_IMPLEMENTED_HEADING_RE.match(heading):
                        report.error(
                            "note.implemented_present",
                            f"{relative}: {heading} is proposal-era; describe current reality instead",
                        )

    validate_archive(root, report, allow_unsealed=allow_unsealed_archive)


def _markdown_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(maxsplit=1)[0]


def maintained_markdown_paths(
    root: Path,
    registry: Mapping[str, Any] | None,
) -> list[Path]:
    values = {root / relative for relative in STANDING_MARKDOWN}
    for lifecycle in ("proposed", "implemented", "rejected"):
        values.update((root / ".agents/notes" / lifecycle).glob("*.md"))
    values.update((root / ".agents/skills").glob("*/SKILL.md"))
    values.update((root / ".agents/skills").glob("*/references/**/*.md"))
    return sorted(values)


def validate_markdown_links(
    root: Path,
    registry: Mapping[str, Any] | None,
    report: VerificationReport,
) -> None:
    for path in maintained_markdown_paths(root, registry):
        relative = _relative(root, path)
        if not path.is_file():
            report.error("markdown.missing", f"maintained Markdown is missing: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(text):
            target = unquote(_markdown_target(match.group(1)))
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            path_text = parsed.path
            if not path_text:
                continue
            resolved = (path.parent / path_text).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                report.error("markdown.link_escape", f"{relative}: link leaves repository: {target}")
                continue
            if not resolved.exists():
                report.error("markdown.link_missing", f"{relative}: missing relative link {target}")


def audit_history(root: Path, report: VerificationReport) -> None:
    for config_path in sorted((root / "configs").rglob("*.yaml")):
        config = _read_yaml(config_path, report, "historical config " + _relative(root, config_path))
        if config is None:
            continue
        for source in _nested_strings(config.get("sources", {})):
            if source.startswith(REPOSITORY_SOURCE_PREFIXES) and not (root / source).is_file():
                report.error(
                    "history.missing_source",
                    f"{_relative(root, config_path)} references missing repository source {source}",
                )


def persistence_paths(
    root: Path,
    registry: Mapping[str, Any] | None,
) -> set[str]:
    paths = {
        ".gitignore",
        ".rgignore",
        REGISTRY_PATH,
        RELEASE_MAP_PATH,
        CURRENT_STATE_PATH,
        BUDGET_PATH,
        "scripts/spec/verify_spec_system.py",
        "tests/test_spec_system.py",
        "scripts/spec/validate_objective_feasibility.py",
        "tests/test_objective_feasibility_contract.py",
        "scripts/spec/check_documentation_change.py",
        "tests/test_documentation_change.py",
        ".githooks/pre-commit",
        ".githooks/pre-push",
        *STANDING_MARKDOWN,
    }
    if (root / ARCHIVE_MANIFEST_PATH).is_file():
        paths.add(ARCHIVE_MANIFEST_PATH)
    for path in maintained_markdown_paths(root, registry):
        paths.add(_relative(root, path))
    # Include supporting resources, including tracked files deleted locally.
    # Git excludes generated caches while retaining untracked authoring work.
    skill_files = _git(
        root, ["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", ".agents/skills/"]
    )
    if skill_files is not None and skill_files.returncode == 0:
        paths.update(filter(None, skill_files.stdout.split("\0")))
    paths.update(
        _relative(root, path)
        for path in (root / ".agents/notes/archived").glob("*.md")
    )
    paths.update(
        _relative(root, path)
        for path in (root / "spec/publication-manifests").glob("*.yaml")
    )
    paths.update(
        _relative(root, path)
        for path in (root / "docs/audits").glob("*.md")
    )
    paths.update(
        _relative(root, path)
        for pattern in ("[0-9]*-pro提问-*.md", "[0-9]*-pro回答-*.md")
        for path in (root / "docs").glob(pattern)
    )
    return {path for path in paths if isinstance(path, str)}


def validate_committed(
    root: Path,
    registry: Mapping[str, Any] | None,
    report: VerificationReport,
) -> None:
    tracked_result = _git(root, ["ls-files", "-z"])
    head_result = _git(root, ["ls-tree", "-rz", "--name-only", "HEAD"])
    changed_result = _git(root, ["diff", "--name-only", "-z", "HEAD", "--"])
    if any(result is None or result.returncode != 0 for result in (tracked_result, head_result, changed_result)):
        report.error("persistence.git", "cannot inspect Git persistence")
        return
    tracked = {value for value in tracked_result.stdout.split("\0") if value}
    in_head = {value for value in head_result.stdout.split("\0") if value}
    changed = {value for value in changed_result.stdout.split("\0") if value}
    # Staged deletions no longer appear in ls-files or the working tree.
    required = persistence_paths(root, registry) | {
        path for path in in_head if path.startswith(".agents/skills/")
    }
    for relative in sorted(required):
        if relative not in tracked:
            report.error("persistence.untracked", f"{relative}: not tracked by Git")
        elif relative not in in_head:
            report.error("persistence.no_head", f"{relative}: not present in HEAD")
        elif relative in changed:
            report.error("persistence.modified", f"{relative}: working tree differs from HEAD")


def verify_repository(
    root: Path,
    *,
    strict_history: bool = False,
    committed: bool = False,
    allow_unsealed_archive: bool = False,
) -> VerificationReport:
    root = root.resolve()
    report = VerificationReport()
    registry = validate_registry(root, report)
    validate_fixed_point_bindings(root, registry, report)
    validate_release_map(root, registry, report)
    validate_current_state(root, registry, report)
    validate_skills(root, report)
    validate_agent_notes(root, report, allow_unsealed_archive=allow_unsealed_archive)
    validate_markdown_links(root, registry, report)
    validate_doc_budgets(root, report)
    if strict_history:
        audit_history(root, report)
    if committed:
        validate_committed(root, registry, report)
    return report


def seal_archive(root: Path) -> VerificationReport:
    root = root.resolve()
    report = VerificationReport()
    validate_agent_notes(root, report, allow_unsealed_archive=True)
    if report.errors:
        return report
    archived = sorted((root / ".agents/notes/archived").glob("*.md"))
    manifest_path = root / ARCHIVE_MANIFEST_PATH
    if not archived and not manifest_path.is_file():
        return report
    if manifest_path.is_file():
        manifest = _archive_manifest(root, report)
        if manifest is None or report.errors:
            return report
        entries = dict(manifest.get("files", {}))
    else:
        entries = {}
    for path in archived:
        relative = _relative(root, path)
        entries.setdefault(relative, hashlib.sha256(path.read_bytes()).hexdigest())
    output = {
        "schema_version": 1,
        "algorithm": "sha256",
        "files": dict(sorted(entries.items())),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root",
    )
    parser.add_argument("--list-budgets", action="store_true", help="Report Unicode character usage")
    parser.add_argument("--strict-history", action="store_true", help="Audit all historical config sources")
    parser.add_argument("--committed", action="store_true", help="Require harness files in clean HEAD")
    parser.add_argument("--seal-archive", action="store_true", help="Append SHA-256 seals for new archived Notes")
    return parser


def _print_axes(report: VerificationReport, *, strict_history: bool, committed: bool) -> None:
    governance_errors = [
        item
        for item in report.errors
        if not item.code.startswith("history.") and not item.code.startswith("persistence.")
    ]
    persistence_errors = [item for item in report.errors if item.code.startswith("persistence.")]
    history_errors = [item for item in report.errors if item.code.startswith("history.")]
    print("governance_structure: " + ("fail" if governance_errors else "pass"))
    print("scientific_contract: not_evaluated")
    if committed:
        print("persistence: " + ("fail" if persistence_errors else "pass"))
    else:
        print("persistence: not_checked")
    if strict_history:
        print("strict_history: " + ("fail" if history_errors else "pass"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_budgets and (args.seal_archive or args.strict_history or args.committed):
        print("[ERROR] cli.mode: --list-budgets is a standalone report mode")
        return 2
    if args.seal_archive and (args.strict_history or args.committed):
        print("[ERROR] cli.mode: --seal-archive cannot be combined with other modes")
        return 2
    root = args.root.resolve()
    if args.list_budgets:
        report = VerificationReport()
        validate_doc_budgets(root, report)
        for row in report.budget_rows:
            status = "ok" if row.used <= row.ceiling else "OVER"
            print(f"BUDGET {status} {row.path}: {row.used}/{row.ceiling} {row.kind}")
        for finding in report.findings:
            print(f"[{finding.severity.upper()}] {finding.code}: {finding.message}")
        return 1 if report.errors else 0
    if args.seal_archive:
        seal_report = seal_archive(root)
        if seal_report.errors:
            report = seal_report
        else:
            report = verify_repository(root)
    else:
        report = verify_repository(
            root,
            strict_history=args.strict_history,
            committed=args.committed,
        )
    for finding in report.findings:
        print(f"[{finding.severity.upper()}] {finding.code}: {finding.message}")
    _print_axes(report, strict_history=args.strict_history, committed=args.committed)
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
