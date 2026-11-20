# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from types import SimpleNamespace

import pytest
from test_product_release import IMAGE, VERSION, commit, git
from test_product_release import release as release  # noqa: PLC0414 -- pytest fixture

from ci import release_pin_pr

REPOSITORY = "NVIDIA/SkillEvaluator"
BRANCH = f"release/pin-v{VERSION}"


@pytest.fixture
def pin_repo(release, tmp_path, monkeypatch):
    repo, source, _, _, candidate = release
    git(repo, "switch", "--detach", source)
    remote = tmp_path / "upstream.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    git(repo, "update-ref", "refs/heads/main", candidate)
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "origin", "main")
    monkeypatch.setattr(release_pin_pr, "UPSTREAM", str(remote))
    monkeypatch.setenv("RELEASE_BUNDLE_APP_SLUG", "release-bot")
    monkeypatch.setattr(release_pin_pr, "gh", lambda *_args: '{"id":765,"login":"release-bot[bot]"}')
    monkeypatch.chdir(repo)
    return repo, candidate, tmp_path / "generated"


def pull(head, *, state="open", merge=None):
    return {
        "number": 12,
        "state": state,
        "base": {"ref": "main", "repo": {"full_name": REPOSITORY}},
        "head": {"ref": BRANCH, "sha": head, "repo": {"full_name": REPOSITORY}},
        "merged_at": "2026-11-20T00:00:00Z" if merge else None,
        "merge_commit_sha": merge,
    }


def test_reuses_signed_bundle_commit_after_interrupted_pr_creation(pin_repo, monkeypatch):
    repo, candidate, generated = pin_repo
    selected = None
    submissions = []
    previous_head = git(repo, "rev-parse", "HEAD")

    def api(path, method="GET", data=None):
        nonlocal selected
        if path.startswith("pulls?"):
            return [selected] if selected else []
        assert path == "pulls" and method == "POST"
        submissions.append(data)
        if len(submissions) == 1:
            raise ValueError("Interrupted before PR creation")
        selected = pull(git(repo, "rev-parse", "refs/remotes/origin/" + BRANCH))
        return selected

    monkeypatch.setattr(release_pin_pr, "api", api)
    with pytest.raises(ValueError, match="Interrupted"):
        release_pin_pr.open_pull(VERSION, candidate, IMAGE, generated)

    ref = f"refs/heads/{BRANCH}"
    original = git(repo, "ls-remote", "--refs", "origin", ref)
    asserted_sha = original.split()[0]
    assert original.endswith(ref)
    assert release_pin_pr.open_pull(VERSION, candidate, IMAGE, generated) == 12
    assert release_pin_pr.open_pull(VERSION, candidate, IMAGE, generated) == 12
    assert len(submissions) == 2
    assert all(data["head"] == BRANCH and data["base"] == "main" for data in submissions)
    assert git(repo, "ls-remote", "--refs", "origin", ref) == original
    assert git(repo, "rev-parse", "HEAD") == previous_head
    assert git(repo, "rev-parse", f"{asserted_sha}^") == candidate
    assert "Signed-off-by: release-bot[bot] <765+release-bot[bot]@users.noreply.github.com>" in git(
        repo, "show", "-s", "--format=%B", asserted_sha
    )


@pytest.mark.parametrize("unexpected", ["extra", "symlink", "modified-wrapper"])
def test_invalid_bundle_cannot_be_pushed(pin_repo, monkeypatch, unexpected):
    repo, candidate, generated = pin_repo
    if unexpected == "extra":
        (generated / "unexpected").write_text("invalid")
    path = generated / ".github/workflows/skill-checks-stable.yml"
    if unexpected == "symlink":
        path.unlink()
        path.symlink_to(repo / ".github/workflows/skill-checks-stable.yml")
    if unexpected == "modified-wrapper":
        path.write_text("# unexpected wrapper\n")
    monkeypatch.setattr(release_pin_pr, "api", lambda *_args, **_kwargs: [])

    with pytest.raises(ValueError, match=r"bundle|stable\.yml"):
        release_pin_pr.open_pull(VERSION, candidate, IMAGE, generated)
    assert not git(repo, "ls-remote", "--refs", "origin", f"refs/heads/{BRANCH}")


def test_wait_returns_merge_from_protected_main(pin_repo, monkeypatch):
    repo, candidate, _ = pin_repo
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "switch", "main")
    git(repo, "merge", "--squash", head)
    merge = commit(repo)
    git(repo, "push", "origin", "main")
    statuses = iter([pull(head), pull(head, state="closed", merge=merge)])
    monkeypatch.setattr(release_pin_pr, "api", lambda _path: next(statuses))
    monkeypatch.setattr(release_pin_pr, "time", SimpleNamespace(monotonic=lambda: 100, sleep=lambda _seconds: None))

    assert release_pin_pr.wait_for_merge(VERSION, candidate, IMAGE, 12, 1) == merge


def test_wait_rejects_commit_unmerged_from_main(pin_repo, monkeypatch):
    repo, candidate, _ = pin_repo
    head = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(release_pin_pr, "api", lambda _path: pull(head, state="closed", merge=head))

    with pytest.raises(ValueError, match="first-parent"):
        release_pin_pr.wait_for_merge(VERSION, candidate, IMAGE, 12, 1)


def test_wait_does_not_accept_a_different_head_repository(pin_repo, monkeypatch):
    repo, candidate, _ = pin_repo
    head = git(repo, "rev-parse", "HEAD")
    result = pull(head)
    result["head"]["repo"]["full_name"] = "elsewhere/SkillEvaluator"
    monkeypatch.setattr(release_pin_pr, "api", lambda _path: result)

    with pytest.raises(ValueError, match="expected upstream"):
        release_pin_pr.wait_for_merge(VERSION, candidate, IMAGE, 12, 1)


def test_wait_stops_at_deadline_without_creating_a_pull_request(pin_repo, monkeypatch):
    repo, candidate, _ = pin_repo
    head = git(repo, "rev-parse", "HEAD")
    timestamps = iter([100, 161])
    monkeypatch.setattr(release_pin_pr, "time", SimpleNamespace(monotonic=lambda: next(timestamps)))
    monkeypatch.setattr(release_pin_pr, "api", lambda _path: pull(head))

    with pytest.raises(ValueError, match="Timed out"):
        release_pin_pr.wait_for_merge(VERSION, candidate, IMAGE, 12, 1)
