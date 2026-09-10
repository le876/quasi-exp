#!/usr/bin/env python3
"""Check documentation affected by a Git index or outgoing ref, without an LLM."""
from __future__ import annotations

import argparse
import posixpath
import re
import subprocess
import sys
import time
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit


LINK = re.compile(r"\[[^\]\n]*\]\(([^)\n]+)\)")


def git(*args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
    ).stdout


def maintained(path: str) -> bool:
    if not path.endswith(".md"):
        return False
    if path.startswith(".agents/notes/archived/"):
        return False
    if path.startswith((".agents/notes/", ".agents/skills/")):
        return True
    parts = PurePosixPath(path).parts
    return (
        path == "docs/protocols/README.md"
        or (len(parts) == 1 and parts[0].isascii())
        or (len(parts) == 2 and parts[0] == "docs" and parts[1].isascii())
    )


def snapshot(target: str | None) -> dict[str, str]:
    args = ("ls-tree", "-rz", "--full-tree", target) if target else ("ls-files", "-s", "-z")
    result = {}
    for row in git(*args).split(b"\0"):
        if not row:
            continue
        meta, path = row.split(b"\t", 1)
        fields = meta.split()
        if target is None and fields[2] != b"0":
            raise ValueError("unmerged index; resolve conflicts before documentation checks")
        result[path.decode()] = fields[2 if target else 1].decode()
    return result


def contents(objects: dict[str, str]) -> dict[str, str]:
    if not objects:
        return {}
    raw = git("cat-file", "--batch", data=("\n".join(objects.values()) + "\n").encode())
    offset = 0
    result = {}
    for path in objects:
        end = raw.index(b"\n", offset)
        header = raw[offset:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"not a documentation blob: {path}")
        size = int(header[2])
        offset = end + 1
        result[path] = raw[offset:offset + size].decode("utf-8")
        offset += size + 1
    return result


def changed_paths(base: str | None, target: str | None) -> set[str]:
    if base is None:
        return set(snapshot(target))
    args = ["diff", "--name-only", "--no-renames", "-z"]
    if target is None:
        args += ["--cached", base]
    else:
        args += [base, target]
    return set(filter(None, git(*args, "--").decode().split("\0")))


def local_links(path: str, text: str):
    # Fenced examples are not active navigation links.
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = None
            continue
        if fence:
            continue
        for match in LINK.finditer(line):
            raw = match[1].strip()
            if not raw:
                continue
            raw = raw[1:raw.index(">")] if raw.startswith("<") and ">" in raw else raw.split()[0]
            parsed = urlsplit(unquote(raw))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            yield posixpath.normpath(posixpath.join(posixpath.dirname(path), parsed.path))


def check(base: str | None, target: str | None) -> tuple[int, set[str]]:
    paths = snapshot(target)
    changed = changed_paths(base, target)
    if not changed:
        return 0, changed
    docs = contents({path: oid for path, oid in paths.items() if maintained(path)})
    available = set(paths)
    for path in paths:
        available.update(str(parent) for parent in PurePosixPath(path).parents)
    errors = []
    related = set()
    for path, text in docs.items():
        links = set(local_links(path, text))
        affected = links & changed
        if path not in changed and (affected or any(p in text for p in changed)):
            related.add(path)
        for link in links if path in changed else affected:
            if link not in available:
                errors.append(f"{path}: target absent from Git snapshot: {link}")
    if any(re.match(r"docs/\d+-pro(?:提问|回答)-", p) or p.startswith("docs/audits/") for p in changed):
        related.add("docs/current-state.md")
    if any(p.startswith(("src/", "configs/", "scripts/analysis/", "scripts/pipelines/")) for p in changed):
        related.update(("spec/registry.yaml", "docs/testing.md"))
    related = (related & available) - changed
    for error in sorted(errors):
        print(f"[ERROR] docs.link: {error}", file=sys.stderr)
    if related:
        print("[REVIEW] 判断是否需同步（可无需修改；不是过期判定）: " + ", ".join(sorted(related)))
    print(f"documentation: {'fail' if errors else 'pass'}; changed={len(changed)}; semantic_freshness=not_evaluated")
    return int(bool(errors)), changed


def governance_change(paths: set[str]) -> bool:
    return any(
        maintained(p)
        or p.startswith(("spec/", ".agents/", ".githooks/", "scripts/spec/", "docs/audits/"))
        or re.match(r"docs/\d+-pro(?:提问|回答)-", p)
        or p in {".gitignore", ".rgignore", "tests/test_spec_system.py", "tests/test_documentation_change.py", "tests/test_objective_feasibility_contract.py"}
        for p in paths
    )


def pre_push() -> int:
    head = git("rev-parse", "HEAD").decode().strip()
    need_verifier = False
    failed = 0
    for line in sys.stdin:
        local_ref, local_sha, _remote_ref, remote_sha = line.split()
        if set(local_sha) == {"0"}:  # Branch deletion has no new documentation.
            continue
        try:
            local_sha = git("rev-parse", local_sha + "^{commit}").decode().strip()
        except subprocess.CalledProcessError:
            print(f"[INFO] {local_ref}: non-commit object; no documentation tree")
            continue
        base = None if set(remote_sha) == {"0"} else remote_sha
        if base:
            try:
                git("cat-file", "-e", base + "^{commit}")
            except subprocess.CalledProcessError:
                print(f"[INFO] {local_ref}: remote base unavailable locally; checking complete target")
                base = None
        print(f"[PUSH] {local_ref} {local_sha[:12]}")
        rc, changed = check(base, local_sha)
        failed |= rc
        # Filtered public snapshots deliberately omit the repository Harness.
        if "spec/registry.yaml" in snapshot(local_sha) and governance_change(changed):
            if local_sha != head:
                print(f"[ERROR] Run governance push checks from a worktree at {local_sha}; current HEAD differs.", file=sys.stderr)
                failed = 1
            else:
                need_verifier = True
    if need_verifier and not failed:
        failed = subprocess.run([sys.executable, "scripts/spec/verify_spec_system.py", "--committed"]).returncode
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-push", action="store_true", help="Read Git pre-push ref updates from stdin")
    args = parser.parse_args()
    started = time.monotonic()
    try:
        if args.pre_push:
            result = pre_push()
        else:
            try:
                base = git("rev-parse", "--verify", "HEAD").decode().strip()
            except subprocess.CalledProcessError:
                base = None
            result, _ = check(base, None)
        print(f"documentation check: {time.monotonic() - started:.3f}s")
        return result
    except (subprocess.CalledProcessError, ValueError, UnicodeError) as exc:
        print(f"[ERROR] documentation check could not read Git snapshot: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
