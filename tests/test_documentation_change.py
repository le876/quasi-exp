from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/spec/check_documentation_change.py"


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def write(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def commit(root):
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    # Isolate fixture repositories from the parent hook/commit environment.
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Doc Test")
    git(tmp_path, "config", "user.email", "doc@example.invalid")
    git(tmp_path, "config", "core.hooksPath", "/dev/null")
    write(tmp_path, "README.md", "# Project\n")
    commit(tmp_path)
    return tmp_path


def run(root, updates=None):
    return subprocess.run(
        [sys.executable, str(CHECKER), *(["--pre-push"] if updates is not None else [])],
        cwd=root, input=updates, text=True, capture_output=True, timeout=15,
    )


def test_unstaged_link_repair_cannot_hide_staged_broken_link(repo):
    write(repo, "README.md", "[Missing](missing.md)\n")
    git(repo, "add", "README.md")
    write(repo, "README.md", "# Repaired only in worktree\n")
    write(repo, "missing.md", "untracked\n")
    result = run(repo)
    assert result.returncode == 1
    assert "missing.md" in result.stderr


def test_staged_target_is_valid_even_if_deleted_only_from_worktree(repo):
    write(repo, "README.md", "[Target](target.md)\n")
    write(repo, "target.md", "# Target\n")
    git(repo, "add", ".")
    (repo / "target.md").unlink()
    assert run(repo).returncode == 0


@pytest.mark.parametrize("operation", ["delete", "rename"])
def test_unchanged_navigation_catches_removed_target(repo, operation):
    write(repo, "README.md", "[Code](src/old.py)\n")
    write(repo, "src/old.py", "pass\n")
    commit(repo)
    if operation == "delete":
        git(repo, "rm", "src/old.py")
    else:
        git(repo, "mv", "src/old.py", "src/new.py")
    result = run(repo)
    assert result.returncode == 1
    assert "README.md" in result.stderr


def test_code_change_only_hints_at_related_note(repo):
    write(repo, "src/model.py", "x = 1\n")
    write(repo, ".agents/notes/implemented/model.md", "Reason for `src/model.py`.\n")
    commit(repo)
    write(repo, "src/model.py", "x = 2\n")
    git(repo, "add", ".")
    result = run(repo)
    assert result.returncode == 0
    assert "[REVIEW]" in result.stdout
    assert "model.md" in result.stdout
    assert "semantic_freshness=not_evaluated" in result.stdout


def test_consultation_change_hints_current_pointer(repo):
    write(repo, "docs/current-state.md", "# Current\n")
    commit(repo)
    write(repo, "docs/33-pro提问-topic.md", "[Historic missing attachment](missing.zip)\n")
    git(repo, "add", ".")
    result = run(repo)
    assert result.returncode == 0
    assert "docs/current-state.md" in result.stdout


def test_archives_and_historical_protocols_are_not_revalidated(repo):
    for path in (".agents/notes/archived/old.md", "docs/protocols/old.md", "原论文.md"):
        write(repo, path, "[Historical](absent.md)\n")
    git(repo, "add", ".")
    assert run(repo).returncode == 0


def test_relative_unicode_spaces_directory_and_examples(repo):
    write(repo, "docs/testing.md", '[File](<../src/中文 file.py>)\n[Dir](../src/)\n[Web](https://example.org/)\n[Anchor](#x)\n```md\n[Example](absent.md)\n```\n')
    write(repo, "src/中文 file.py", "pass\n")
    git(repo, "add", ".")
    assert run(repo).returncode == 0


def test_initial_commit(repo):
    git(repo, "checkout", "--orphan", "initial")
    write(repo, "README.md", "# Initial\n")
    git(repo, "add", ".")
    assert run(repo).returncode == 0


def test_multiple_outgoing_refs_use_their_own_trees(repo):
    base = git(repo, "rev-parse", "HEAD")
    write(repo, "README.md", "[Missing](absent.md)\n")
    broken = commit(repo)
    write(repo, "README.md", "# Repaired\n")
    repaired = commit(repo)
    result = run(repo, f"refs/heads/old {broken} refs/heads/old {base}\nrefs/heads/main {repaired} refs/heads/main {base}\n")
    assert result.returncode == 1
    assert "absent.md" in result.stderr
    assert result.stdout.count("[PUSH]") == 2


def test_new_ref_missing_remote_and_ref_deletion(repo):
    head = git(repo, "rev-parse", "HEAD")
    zero = "0" * 40
    for remote in (zero, "a" * 40):
        assert run(repo, f"refs/heads/new {head} refs/heads/new {remote}\n").returncode == 0
    assert run(repo, f"(delete) {zero} refs/heads/old {head}\n").returncode == 0


def test_governance_push_runs_verifier_once_only_for_head(repo):
    write(repo, "spec/registry.yaml", "fixture\n")
    write(repo, "scripts/spec/verify_spec_system.py", "from pathlib import Path\np=Path('calls')\np.write_text(p.read_text()+'x' if p.exists() else 'x')\n")
    source = commit(repo)
    zero = "0" * 40
    updates = f"refs/heads/main {source} refs/heads/main {zero}\nrefs/heads/alias {source} refs/heads/alias {zero}\n"
    assert run(repo, updates).returncode == 0
    assert (repo / "calls").read_text() == "x"
    write(repo, "README.md", "# New head\n")
    commit(repo)
    result = run(repo, updates)
    assert result.returncode == 1
    assert "current HEAD differs" in result.stderr


def test_real_git_commit_invokes_hook(repo):
    write(repo, "scripts/spec/check_documentation_change.py", CHECKER.read_text())
    hook_dir = repo / ".githooks"
    hook_dir.mkdir()
    hook = hook_dir / "pre-commit"
    shutil.copyfile(ROOT / ".githooks/pre-commit", hook)
    hook.chmod(0o755)
    git(repo, "config", "core.hooksPath", ".githooks")
    write(repo, "README.md", "[Missing](absent.md)\n")
    git(repo, "add", ".")
    result = subprocess.run(["git", "commit", "-m", "must fail"], cwd=repo, capture_output=True, text=True)
    assert result.returncode != 0
    assert "docs.link" in result.stderr


def test_archived_note_push_still_requires_seal_verification(repo):
    write(repo, "spec/registry.yaml", "fixture\n")
    write(repo, "scripts/spec/verify_spec_system.py", "raise SystemExit(7)\n")
    base = commit(repo)
    write(repo, ".agents/notes/archived/old.md", "changed frozen content\n")
    head = commit(repo)
    result = run(repo, f"refs/heads/main {head} refs/heads/main {base}\n")
    assert result.returncode == 7


def test_annotated_tag_at_head_uses_commit_identity(repo):
    write(repo, "spec/registry.yaml", "fixture\n")
    write(repo, "scripts/spec/verify_spec_system.py", "print('VERIFIED')\n")
    commit(repo)
    git(repo, "tag", "-a", "v1", "-m", "version")
    tag = git(repo, "rev-parse", "v1")
    result = run(repo, f"refs/tags/v1 {tag} refs/tags/v1 {'0' * 40}\n")
    assert result.returncode == 0
    assert "VERIFIED" in result.stdout
