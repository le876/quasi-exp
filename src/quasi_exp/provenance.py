"""Fail-closed provenance primitives for immutable experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


ARTIFACT_MANIFEST_TYPE = "quasi_exp.artifact_manifest"
SOURCE_FIXED_POINT_TYPE = "quasi_exp.source_fixed_point"
PROTOCOL_SOURCE_SNAPSHOT_TYPE = "quasi_exp.protocol_source_snapshot"
FAMILY_DISPOSITION_TYPE = "quasi_exp.family_disposition"
SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40,64}$")


class ProvenanceError(RuntimeError):
    """Base class for provenance failures."""


class ManifestSchemaError(ProvenanceError):
    """Raised when a manifest does not satisfy its declared schema."""


class ArtifactTreeError(ProvenanceError):
    """Raised when an artifact tree cannot be inventoried safely."""


class SourceLockError(ProvenanceError):
    """Raised when execution source differs from its fixed point."""


class FamilyDispositionError(ProvenanceError):
    """Raised when a development family is reused as a sealed family."""


@dataclass(frozen=True)
class FileRecord:
    """Canonical identity for one project-relative file."""

    path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SourceLockState:
    """Verified source state used to bind all stages of one run."""

    source_root: Path
    git_sha: str
    manifest_path: Path
    manifest_relative_path: str
    manifest_sha256: str
    execution_sources: tuple[FileRecord, ...]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestSchemaError(f"{field} must be a non-empty string")
    if "\\" in value:
        raise ManifestSchemaError(f"{field} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ManifestSchemaError(f"{field} must be a normalized relative path")
    normalized = path.as_posix()
    if normalized != value:
        raise ManifestSchemaError(f"{field} must be normalized")
    return normalized


def _sha256_value(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ManifestSchemaError(f"{field} must be a lowercase SHA256")
    return value


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestSchemaError(f"{field} must be a nonnegative integer")
    return value


def _file_records(
    payload: Any,
    *,
    field: str,
    require_sorted: bool = True,
) -> tuple[FileRecord, ...]:
    if not isinstance(payload, list):
        raise ManifestSchemaError(f"{field} must be a list")
    records: list[FileRecord] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ManifestSchemaError(f"{field}[{index}] must be an object")
        records.append(
            FileRecord(
                path=_normalized_relative_path(
                    item.get("path"), field=f"{field}[{index}].path"
                ),
                size_bytes=_nonnegative_int(
                    item.get("size_bytes"),
                    field=f"{field}[{index}].size_bytes",
                ),
                sha256=_sha256_value(
                    item.get("sha256"),
                    field=f"{field}[{index}].sha256",
                ),
            )
        )
    paths = [record.path for record in records]
    if len(paths) != len(set(paths)):
        raise ManifestSchemaError(f"{field} contains duplicate paths")
    if require_sorted and paths != sorted(paths):
        raise ManifestSchemaError(f"{field} must be sorted by path")
    return tuple(records)


def collect_tree(root: Path) -> tuple[FileRecord, ...]:
    """Return a canonical, symlink-free inventory for ``root``."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")
    records: list[FileRecord] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ArtifactTreeError(
                f"symlink is not allowed in an immutable artifact: {path}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(resolved).as_posix()
        records.append(
            FileRecord(
                path=relative,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return tuple(records)


def canonical_tree_sha256(records: Sequence[FileRecord]) -> str:
    """Hash normalized path, byte size, and content digest in path order."""

    digest = sha256()
    for record in sorted(records, key=lambda item: item.path):
        digest.update(
            (
                f"{record.path}\0{record.size_bytes}\0"
                f"{record.sha256}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def tree_summary(records: Sequence[FileRecord]) -> dict[str, Any]:
    ordered = tuple(sorted(records, key=lambda item: item.path))
    return {
        "canonicalization": (
            "SHA256 over UTF-8 '<path>\\0<size_bytes>\\0<sha256>\\n' "
            "records sorted by normalized POSIX path"
        ),
        "file_count": len(ordered),
        "total_bytes": sum(record.size_bytes for record in ordered),
        "tree_sha256": canonical_tree_sha256(ordered),
        "files": [record.as_dict() for record in ordered],
    }


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestSchemaError(f"cannot read JSON manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestSchemaError("manifest root must be an object")
    return payload


def validate_artifact_manifest(payload: Mapping[str, Any]) -> tuple[FileRecord, ...]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestSchemaError("unsupported artifact manifest schema_version")
    if payload.get("manifest_type") != ARTIFACT_MANIFEST_TYPE:
        raise ManifestSchemaError("invalid artifact manifest_type")
    if not isinstance(payload.get("artifact_id"), str) or not payload["artifact_id"]:
        raise ManifestSchemaError("artifact_id must be a non-empty string")
    _normalized_relative_path(payload.get("logical_root"), field="logical_root")
    seal = payload.get("seal")
    if not isinstance(seal, Mapping):
        raise ManifestSchemaError("seal must be an object")
    if seal.get("status") != "retrospectively_sealed_current_state":
        raise ManifestSchemaError("seal.status must describe the retrospective seal")
    limitations = seal.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise ManifestSchemaError("seal.limitations must be a non-empty list")
    tree = payload.get("tree")
    if not isinstance(tree, Mapping):
        raise ManifestSchemaError("tree must be an object")
    records = _file_records(tree.get("files"), field="tree.files")
    _nonnegative_int(tree.get("file_count"), field="tree.file_count")
    _nonnegative_int(tree.get("total_bytes"), field="tree.total_bytes")
    _sha256_value(tree.get("tree_sha256"), field="tree.tree_sha256")
    lineage = payload.get("runner_lineage")
    if not isinstance(lineage, Mapping):
        raise ManifestSchemaError("runner_lineage must be an object")
    for stage in ("protocol", "teacher", "student", "visualize", "summary"):
        item = lineage.get(stage)
        if not isinstance(item, Mapping):
            raise ManifestSchemaError(f"runner_lineage.{stage} must be an object")
        _sha256_value(item.get("sha256"), field=f"runner_lineage.{stage}.sha256")
    fixed_point = payload.get("implementation_fixed_point")
    if not isinstance(fixed_point, Mapping):
        raise ManifestSchemaError("implementation_fixed_point must be an object")
    _sha256_value(
        fixed_point.get("source_fixed_point_sha256"),
        field="implementation_fixed_point.source_fixed_point_sha256",
    )
    disposition = payload.get("family_disposition")
    if not isinstance(disposition, Mapping):
        raise ManifestSchemaError("family_disposition must be an object")
    _sha256_value(
        disposition.get("sha256"), field="family_disposition.sha256"
    )
    return records


def verify_artifact_tree(
    manifest: Mapping[str, Any],
    artifact_root: Path,
) -> list[str]:
    """Return exact-set verification errors; an empty list means success."""

    expected = validate_artifact_manifest(manifest)
    actual = collect_tree(artifact_root)
    tree = manifest["tree"]
    errors: list[str] = []
    expected_by_path = {record.path: record for record in expected}
    actual_by_path = {record.path: record for record in actual}
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    extra = sorted(set(actual_by_path) - set(expected_by_path))
    if missing:
        errors.append("missing files: " + ", ".join(missing))
    if extra:
        errors.append("extra files: " + ", ".join(extra))
    for path in sorted(set(expected_by_path) & set(actual_by_path)):
        wanted = expected_by_path[path]
        found = actual_by_path[path]
        if wanted.size_bytes != found.size_bytes:
            errors.append(
                f"size mismatch: {path}: expected {wanted.size_bytes}, "
                f"found {found.size_bytes}"
            )
        if wanted.sha256 != found.sha256:
            errors.append(
                f"sha256 mismatch: {path}: expected {wanted.sha256}, "
                f"found {found.sha256}"
            )
    actual_count = len(actual)
    actual_total = sum(record.size_bytes for record in actual)
    actual_tree_sha = canonical_tree_sha256(actual)
    if int(tree["file_count"]) != len(expected):
        errors.append("manifest file_count disagrees with manifest file entries")
    if int(tree["total_bytes"]) != sum(record.size_bytes for record in expected):
        errors.append("manifest total_bytes disagrees with manifest file entries")
    if str(tree["tree_sha256"]) != canonical_tree_sha256(expected):
        errors.append("manifest tree_sha256 disagrees with manifest file entries")
    if int(tree["file_count"]) != actual_count:
        errors.append(
            f"file_count mismatch: expected {tree['file_count']}, "
            f"found {actual_count}"
        )
    if int(tree["total_bytes"]) != actual_total:
        errors.append(
            f"total_bytes mismatch: expected {tree['total_bytes']}, "
            f"found {actual_total}"
        )
    if str(tree["tree_sha256"]) != actual_tree_sha:
        errors.append(
            f"tree_sha256 mismatch: expected {tree['tree_sha256']}, "
            f"found {actual_tree_sha}"
        )
    return errors


def _git(source_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SourceLockError(
            f"git {' '.join(arguments)} failed with "
            f"{completed.returncode}: {detail}"
        )
    return completed.stdout


def _validate_source_fixed_point(
    payload: Mapping[str, Any],
) -> tuple[FileRecord, ...]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestSchemaError("unsupported source fixed-point schema_version")
    if payload.get("manifest_type") != SOURCE_FIXED_POINT_TYPE:
        raise ManifestSchemaError("invalid source fixed-point manifest_type")
    formal = payload.get("formal_config")
    if not isinstance(formal, Mapping):
        raise ManifestSchemaError("formal_config must be an object")
    _normalized_relative_path(
        formal.get("entrypoint"), field="formal_config.entrypoint"
    )
    records = _file_records(
        payload.get("execution_sources"),
        field="execution_sources",
    )
    if not records:
        raise ManifestSchemaError("execution_sources must not be empty")
    return records


def _tracked(source_root: Path, relative_path: str) -> bool:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "ls-files",
            "--error-unmatch",
            "--",
            relative_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def preflight_source_fixed_point(
    source_root: Path,
    manifest_path: Path,
    *,
    requested_config: Path | None = None,
) -> SourceLockState:
    """Require a clean Git worktree whose declared files match the lock."""

    root = source_root.resolve()
    if not root.is_dir():
        raise SourceLockError(f"source root does not exist: {source_root}")
    inside = _git(root, "rev-parse", "--is-inside-work-tree").strip()
    if inside != "true":
        raise SourceLockError("execution source is not a Git worktree")
    top = Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != root:
        raise SourceLockError(
            f"execution source root {root} is not Git toplevel {top}"
        )
    manifest = manifest_path.resolve()
    try:
        manifest_relative = manifest.relative_to(root).as_posix()
    except ValueError as error:
        raise SourceLockError("source fixed-point manifest is outside source root") from error
    if not _tracked(root, manifest_relative):
        raise SourceLockError(
            f"source fixed-point manifest is not tracked: {manifest_relative}"
        )
    status_before = _git(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status_before:
        raise SourceLockError(
            "execution worktree is dirty; git status --porcelain is not empty: "
            + status_before.strip().replace("\n", "; ")
        )
    payload = load_json_object(manifest)
    records = _validate_source_fixed_point(payload)
    if requested_config is not None:
        expected = (
            root / str(payload["formal_config"]["entrypoint"])
        ).resolve()
        if requested_config.resolve() != expected:
            raise SourceLockError(
                f"config is not the fixed formal entrypoint: "
                f"expected {expected}, found {requested_config.resolve()}"
            )
    mismatches: list[str] = []
    for record in records:
        if not _tracked(root, record.path):
            mismatches.append(f"untracked source: {record.path}")
            continue
        path = root / record.path
        if not path.is_file() or path.is_symlink():
            mismatches.append(f"missing or non-regular source: {record.path}")
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != record.size_bytes:
            mismatches.append(
                f"size mismatch: {record.path}: "
                f"expected {record.size_bytes}, found {size}"
            )
        if digest != record.sha256:
            mismatches.append(
                f"sha256 mismatch: {record.path}: "
                f"expected {record.sha256}, found {digest}"
            )
    if mismatches:
        raise SourceLockError("; ".join(mismatches))
    git_sha = _git(root, "rev-parse", "HEAD").strip()
    if not _GIT_SHA_RE.fullmatch(git_sha):
        raise SourceLockError(f"invalid Git SHA: {git_sha}")
    status_after = _git(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    git_sha_after = _git(root, "rev-parse", "HEAD").strip()
    if status_after or git_sha_after != git_sha:
        raise SourceLockError("execution source changed during provenance preflight")
    return SourceLockState(
        source_root=root,
        git_sha=git_sha,
        manifest_path=manifest,
        manifest_relative_path=manifest_relative,
        manifest_sha256=sha256_file(manifest),
        execution_sources=records,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _snapshot_records(state: SourceLockState) -> tuple[FileRecord, ...]:
    manifest_record = FileRecord(
        path=state.manifest_relative_path,
        size_bytes=state.manifest_path.stat().st_size,
        sha256=state.manifest_sha256,
    )
    records = (manifest_record, *state.execution_sources)
    by_path = {record.path: record for record in records}
    if len(by_path) != len(records):
        raise SourceLockError(
            "source fixed-point manifest must not list itself as an execution source"
        )
    return tuple(sorted(records, key=lambda item: item.path))


def _refresh_source_lock_state(state: SourceLockState) -> SourceLockState:
    refreshed = preflight_source_fixed_point(
        state.source_root,
        state.manifest_path,
    )
    if refreshed != state:
        raise SourceLockError(
            "execution source changed after the fixed-point preflight"
        )
    return refreshed


def write_protocol_source_snapshot(
    state: SourceLockState,
    output_root: Path,
) -> dict[str, Any]:
    """Atomically copy the verified execution closure into a new artifact."""

    _refresh_source_lock_state(state)
    stage = output_root.resolve() / "00_protocol"
    destination = stage / "source_snapshot"
    if destination.exists():
        raise SourceLockError(
            f"protocol source snapshot already exists: {destination}"
        )
    stage.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".source_snapshot.", dir=str(stage))
    )
    records = _snapshot_records(state)
    try:
        for record in records:
            source = state.source_root / record.path
            copied = temporary / record.path
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
            if (
                copied.stat().st_size != record.size_bytes
                or sha256_file(copied) != record.sha256
            ):
                raise SourceLockError(
                    f"source changed while copying protocol snapshot: {record.path}"
                )
        _refresh_source_lock_state(state)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": PROTOCOL_SOURCE_SNAPSHOT_TYPE,
            "git_sha": state.git_sha,
            "source_fixed_point_path": state.manifest_relative_path,
            "source_fixed_point_sha256": state.manifest_sha256,
            "files": [record.as_dict() for record in records],
        }
        _write_json(temporary / "source_lock.json", payload)
        os.replace(temporary, destination)
        return payload
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_protocol_source_snapshot(
    state: SourceLockState,
    output_root: Path,
) -> dict[str, Any]:
    """Bind a later stage to the exact Git/source state frozen by protocol."""

    _refresh_source_lock_state(state)
    snapshot = output_root.resolve() / "00_protocol/source_snapshot"
    lock_path = snapshot / "source_lock.json"
    payload = load_json_object(lock_path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SourceLockError("unsupported protocol source snapshot schema")
    if payload.get("manifest_type") != PROTOCOL_SOURCE_SNAPSHOT_TYPE:
        raise SourceLockError("invalid protocol source snapshot manifest_type")
    if payload.get("git_sha") != state.git_sha:
        raise SourceLockError(
            "Git SHA differs from the protocol source snapshot"
        )
    if payload.get("source_fixed_point_path") != state.manifest_relative_path:
        raise SourceLockError(
            "source fixed-point path differs from the protocol snapshot"
        )
    if payload.get("source_fixed_point_sha256") != state.manifest_sha256:
        raise SourceLockError(
            "source fixed-point manifest differs from the protocol snapshot"
        )
    try:
        locked = _file_records(payload.get("files"), field="files")
    except ManifestSchemaError as error:
        raise SourceLockError(str(error)) from error
    expected = _snapshot_records(state)
    if locked != expected:
        raise SourceLockError(
            "declared source file set differs from the protocol snapshot"
        )
    expected_paths = {record.path for record in expected}
    actual_paths: set[str] = set()
    for path in sorted(snapshot.rglob("*")):
        if path.is_symlink():
            raise SourceLockError(
                f"symlink in protocol source snapshot: {path}"
            )
        if path.is_file() and path != lock_path:
            actual_paths.add(path.relative_to(snapshot).as_posix())
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise SourceLockError(
            f"protocol source snapshot file set changed; "
            f"missing={missing}; extra={extra}"
        )
    for record in expected:
        copied = snapshot / record.path
        if (
            copied.stat().st_size != record.size_bytes
            or sha256_file(copied) != record.sha256
        ):
            raise SourceLockError(
                f"protocol source snapshot content changed: {record.path}"
            )
    return payload


def validate_family_disposition(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestSchemaError("unsupported family disposition schema_version")
    if payload.get("manifest_type") != FAMILY_DISPOSITION_TYPE:
        raise ManifestSchemaError("invalid family disposition manifest_type")
    items = payload.get("development_boundary_families")
    if not isinstance(items, list):
        raise ManifestSchemaError(
            "development_boundary_families must be a list"
        )
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    fingerprints: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ManifestSchemaError(
                f"development_boundary_families[{index}] must be an object"
            )
        family_id = item.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            raise ManifestSchemaError(
                f"development_boundary_families[{index}].family_id is invalid"
            )
        fingerprint = _sha256_value(
            item.get("family_fingerprint"),
            field=(
                f"development_boundary_families[{index}]."
                "family_fingerprint"
            ),
        )
        if item.get("final_sealed_test_eligible") is not False:
            raise ManifestSchemaError(
                "development boundary families must set "
                "final_sealed_test_eligible=false"
            )
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ManifestSchemaError(
                f"development_boundary_families[{index}].reason is invalid"
            )
        if family_id in ids or fingerprint in fingerprints:
            raise ManifestSchemaError(
                "development boundary family IDs and fingerprints must be unique"
            )
        ids.add(family_id)
        fingerprints.add(fingerprint)
        normalized.append(dict(item))
    return tuple(normalized)


def load_family_disposition(path: Path) -> dict[str, Any]:
    payload = load_json_object(path)
    validate_family_disposition(payload)
    return payload


def assert_final_sealed_family_eligible(
    family_id: str,
    family_fingerprint: str,
    disposition: Mapping[str, Any],
) -> None:
    """Reject reuse by either identity, including fingerprint-preserving renames."""

    entries = validate_family_disposition(disposition)
    id_matches = [
        item for item in entries if item["family_id"] == str(family_id)
    ]
    fingerprint_matches = [
        item
        for item in entries
        if item["family_fingerprint"] == str(family_fingerprint)
    ]
    if id_matches or fingerprint_matches:
        matched_by: list[str] = []
        if id_matches:
            matched_by.append("family_id")
        if fingerprint_matches:
            matched_by.append("family_fingerprint")
        raise FamilyDispositionError(
            f"family is development/boundary evidence and is not eligible "
            f"for a final sealed test; matched by {', '.join(matched_by)}"
        )
