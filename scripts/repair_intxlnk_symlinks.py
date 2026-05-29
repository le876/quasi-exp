#!/usr/bin/env python3
"""Repair malformed IntxLNK placeholder files into real symlinks.

Some broken copy/mount workflows may turn symlinks inside Conda envs into
tiny regular files that start with b"IntxLNK". This script decodes those
placeholders and restores actual symlinks when targets exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def decode_intx_target(raw: bytes) -> str:
    if not raw.startswith(b"IntxLNK"):
        return ""
    # Format observed: b"IntxLNK" + b"\x01" + UTF-16LE target text
    payload = raw[8:]
    return payload.decode("utf-16le", errors="ignore").rstrip("\x00")


def repair_libdir(libdir: Path, dry_run: bool) -> tuple[int, int, int]:
    scanned = 0
    repaired = 0
    skipped_missing = 0

    for so_path in libdir.glob("lib*.so*"):
        scanned += 1
        try:
            if so_path.is_symlink():
                continue
            if so_path.stat().st_size > 512:
                continue
            raw = so_path.read_bytes()
        except OSError:
            continue

        target_name = decode_intx_target(raw)
        if not target_name:
            continue

        target_path = libdir / target_name
        if not target_path.exists():
            skipped_missing += 1
            print(f"SKIP missing target: {so_path.name} -> {target_name}")
            continue

        print(f"FIX {so_path.name} -> {target_name}")
        if not dry_run:
            so_path.unlink()
            so_path.symlink_to(target_name)
        repaired += 1

    return scanned, repaired, skipped_missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair IntxLNK placeholder files in a conda env lib dir."
    )
    parser.add_argument(
        "env_prefix",
        nargs="?",
        default="/mnt/ML_projects/conda_envs/dante_env",
        help="Conda env prefix path (default: /mnt/ML_projects/conda_envs/dante_env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be changed.",
    )
    args = parser.parse_args()

    libdir = Path(args.env_prefix) / "lib"
    if not libdir.is_dir():
        print(f"ERROR: lib dir not found: {libdir}")
        return 2

    scanned, repaired, skipped_missing = repair_libdir(libdir, args.dry_run)
    print(
        f"done scanned={scanned} repaired={repaired} missing_target={skipped_missing} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
