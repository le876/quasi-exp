#!/usr/bin/env python3
"""Verify an artifact against an immutable, Git-tracked manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))

from quasi_exp.provenance import (  # noqa: E402
    ArtifactTreeError,
    ManifestSchemaError,
    load_json_object,
    validate_artifact_manifest,
    verify_artifact_tree,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    if not manifest_path.is_file():
        print(
            f"invalid manifest path: {manifest_path}",
            file=sys.stderr,
        )
        return 2
    if not artifact_root.is_dir():
        print(
            f"invalid artifact root: {artifact_root}",
            file=sys.stderr,
        )
        return 2
    try:
        manifest = load_json_object(manifest_path)
        records = validate_artifact_manifest(manifest)
    except ManifestSchemaError as error:
        print(f"invalid manifest schema: {error}", file=sys.stderr)
        return 2
    try:
        errors = verify_artifact_tree(manifest, artifact_root)
    except ArtifactTreeError as error:
        errors = [str(error)]
    if errors:
        print(
            json.dumps(
                {
                    "artifact_root": str(artifact_root),
                    "error_count": len(errors),
                    "errors": errors,
                    "manifest": str(manifest_path),
                    "verified": False,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "artifact_root": str(artifact_root),
                "file_count": len(records),
                "manifest": str(manifest_path),
                "total_bytes": sum(record.size_bytes for record in records),
                "tree_sha256": manifest["tree"]["tree_sha256"],
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
