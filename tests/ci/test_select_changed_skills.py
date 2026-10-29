# SPDX-License-Identifier: Apache-2.0
"""Changed skill selection uses real Git history, not shaped diff mocks."""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "changed_selection", Path(__file__).parents[2] / "ci/select_changed_skills.py"
)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def git(repo, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgSign=false", *arguments],
        text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
        stderr=subprocess.DEVNULL,
    ).strip()


def commit(repo):
    git(repo, "add", "-A")
    git(repo, "commit", "--allow-empty", "-m", "fixture")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "checkout with spaces"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Selection Test")
    git(repo, "config", "user.email", "selection@example.invalid")
    for name in ("first skill", "second"):
        skill = repo / "skills" / name
        (skill / "scripts").mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"# {name}\n")
        (skill / "scripts/helper.py").write_text("print('fixture')\n")
    (repo / "README.md").write_text("Unrelated repository documentation\n")
    return repo, commit(repo)


def select(repository, base, head, roots="skills"):
    targets, receipt = selector.select_changed_skills(repository, roots, base, head)
    assert [path.relative_to(repository).as_posix() for path in targets] == receipt["selected"]
    return receipt


def test_unrelated_commit_is_explicit_empty_selection(repository):
    repo, base = repository
    (repo / "README.md").write_text("Unrelated update\n")
    head = commit(repo)
    receipt = select(repo, base, head)
    assert receipt == {
        "mode": "changed",
        "base": base,
        "base_ref": base,
        "head": head,
        "discovered": 2,
        "selected": [],
        "deleted": [],
        "changed_files": 0,
    }


def test_changed_skill_path_with_del_is_rejected_before_scanning(repository):
    repo, base = repository
    path = repo / "skills" / "del\x7fskill"
    path.mkdir()
    (path / "SKILL.md").write_text("# skill\n")
    head = commit(repo)
    with pytest.raises(selector.SelectionError):
        select(repo, base, head)


def test_configured_root_with_del_is_rejected(repository):
    repo, _ = repository
    path = repo / "skills" / "del\x7fskill"
    path.mkdir()
    (path / "SKILL.md").write_text("# skill\n")
    head = commit(repo)
    with pytest.raises(selector.SelectionError):
        select(repo, head, head, "skills/del\x7fskill")


@pytest.mark.parametrize("operation", ["modify", "delete-file", "rename-file"])
def test_changes_within_skill_select_only_that_skill(repository, operation):
    repo, base = repository
    path = repo / "skills/first skill/scripts/helper.py"
    if operation == "modify":
        path.write_text("print('changed')\n")
    elif operation == "delete-file":
        path.unlink()
    else:
        git(repo, "mv", "skills/first skill/scripts/helper.py", "skills/first skill/scripts/renamed helper.py")
    head = commit(repo)
    assert select(repo, base, head, "skills\nskills/first skill\nskills")["selected"] == ["skills/first skill"]


def test_whole_skill_rename_selects_destination_and_records_removed_source(repository):
    repo, base = repository
    git(repo, "mv", "skills/first skill", "skills/renamed skill")
    head = commit(repo)
    receipt = select(repo, base, head)
    assert receipt["selected"] == ["skills/renamed skill"]
    assert receipt["deleted"] == ["skills/first skill"]


def test_manifest_removal_with_content_left_is_error_not_noop(repository):
    repo, base = repository
    (repo / "skills/first skill/SKILL.md").unlink()
    head = commit(repo)
    with pytest.raises(selector.SelectionError, match=r"Deleted SKILL\.md left tracked content") as error:
        select(repo, base, head)
    assert error.value.selection["deleted"] == []


def test_complete_skill_deletion_is_recorded_even_when_its_configured_root_disappears(repository):
    repo, base = repository
    shutil.rmtree(repo / "skills/first skill")
    head = commit(repo)
    receipt = select(repo, base, head, "skills/first skill")
    assert receipt["selected"] == [] and receipt["deleted"] == ["skills/first skill"]
    assert receipt["discovered"] == 0


def test_nested_manifests_do_not_add_unchanged_descendant_targets(repository):
    repo, _ = repository
    nested = repo / "skills/first skill/nested"
    nested.mkdir()
    (nested / "SKILL.md").write_text("# Nested\n")
    base = commit(repo)
    (repo / "skills/first skill/SKILL.md").write_text("# Changed parent manifest\n")
    head = commit(repo)
    assert select(repo, base, head)["selected"] == ["skills/first skill"]
    (nested / "SKILL.md").write_text("# Changed nested content\n")
    newer = commit(repo)
    assert select(repo, head, newer)["selected"] == ["skills/first skill", "skills/first skill/nested"]


def test_configured_roots_are_literal_not_git_pathspec_globs(repository):
    repo, _ = repository
    folder = repo / "skills/[literal]"
    folder.mkdir()
    (folder / "SKILL.md").write_text("# Literal\n")
    base = commit(repo)
    (folder / "SKILL.md").write_text("# Literal changed\n")
    head = commit(repo)
    assert select(repo, base, head, "skills/[literal]")["selected"] == ["skills/[literal]"]


@pytest.mark.parametrize("dirty", ["modified", "untracked", "ignored"])
def test_checkout_content_must_match_supplied_head(repository, dirty):
    repo, base = repository
    if dirty == "modified":
        (repo / "skills/first skill/SKILL.md").write_text("not committed\n")
    elif dirty == "untracked":
        (repo / "skills/first skill/untracked.txt").write_text("not committed\n")
    else:
        (repo / ".git/info/exclude").write_text("skills/first skill/cache.txt\n")
        (repo / "skills/first skill/cache.txt").write_text("ignored content\n")
    with pytest.raises(selector.SelectionError, match="checkout must match"):
        select(repo, base, base)


@pytest.mark.parametrize("base", ["", "main", "--help", "0" * 40])
def test_unknown_or_non_commit_base_never_falls_back_to_full_scan(repository, base):
    repo, head = repository
    with pytest.raises(selector.SelectionError):
        select(repo, base, head)


def test_mismatched_head_is_explicit_error(repository):
    repo, base = repository
    (repo / "README.md").write_text("newer\n")
    commit(repo)
    with pytest.raises(selector.SelectionError, match="HEAD does not match"):
        select(repo, base, base)


def test_selected_skill_limit_and_git_output_limit_are_enforced(repository, monkeypatch):
    repo, base = repository
    for name in ("first skill", "second"):
        (repo / "skills" / name / "SKILL.md").write_text("changed\n")
    head = commit(repo)
    monkeypatch.setattr(selector, "MAX_SELECTED_SKILLS", 1)
    with pytest.raises(selector.SelectionError, match="selected skills"):
        select(repo, base, head)
    monkeypatch.setattr(selector, "MAX_GIT_BYTES", 1)
    with pytest.raises(selector.SelectionError, match="output limit"):
        select(repo, base, head)


def test_repository_diff_commands_and_fsmonitor_are_not_executed(repository, tmp_path, monkeypatch):
    repo, base = repository
    (repo / "skills/first skill/SKILL.md").write_text("changed\n")
    head = commit(repo)
    marker = tmp_path / "hook-executed"
    script = tmp_path / "untrusted-command.sh"
    script.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    script.chmod(0o700)
    git(repo, "config", "core.fsmonitor", str(script))
    git(repo, "config", "diff.external", str(script))
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", str(script))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(script))
    assert select(repo, base, head)["selected"] == ["skills/first skill"]
    assert not marker.exists()


def test_diverged_target_branch_does_not_select_base_only_skill_changes(repository):
    repo, common = repository
    git(repo, "switch", "-c", "feature")
    (repo / "skills/first skill/SKILL.md").write_text("# Feature change\n")
    head = commit(repo)
    git(repo, "switch", "-c", "advanced-target", common)
    (repo / "skills/second/SKILL.md").write_text("# Target-only change\n")
    base_ref = commit(repo)
    git(repo, "switch", "feature")
    receipt = select(repo, base_ref, head)
    assert receipt["selected"] == ["skills/first skill"]
    assert receipt["base"] == common and receipt["base_ref"] == base_ref


def test_unrelated_histories_do_not_fall_back_to_full_scan(repository):
    repo, base = repository
    git(repo, "switch", "--orphan", "unrelated")
    git(repo, "rm", "-rf", "--ignore-unmatch", ".")
    folder = repo / "skills/new"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("# New history\n")
    head = commit(repo)
    with pytest.raises(selector.SelectionError, match="common merge base"):
        select(repo, base, head)
