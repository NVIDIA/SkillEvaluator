# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current-head and ownership boundaries for trusted PR feedback."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("comment_report", Path(__file__).parents[2] / "ci/comment_report.py")
feedback = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(feedback)
BOT = "github-actions[bot]"
SHA = "a" * 40
REPOSITORY = "owner/consumer"


def prepare(directory, **changes):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_url": "https://github.com/owner/consumer/actions/runs/123",
        "pr_number": "1",
        "source_head_sha": SHA,
        "is_current": "true",
        "run_attempt": "1",
        **changes,
    }
    (directory / "publication.json").write_text(json.dumps(payload), encoding="utf-8")
    (directory / "pr-comment.md").write_text(
        "Failed check: @somebody\n[Full report](https://owner.github.io/consumer/)", encoding="utf-8"
    )


def comment(identifier, body, *, login=BOT, user_type="Bot"):
    return {
        "id": identifier,
        "body": body,
        "user": {"login": login, "type": user_type},
        "html_url": f"https://github.com/owner/consumer/pull/1#issuecomment-{identifier}",
    }


class FakeAPI:
    def __init__(self, *, comments=None, sha=SHA, state="open", head_repo=REPOSITORY):
        self.comments = comments or []
        self.sha, self.state, self.head_repo = sha, state, head_repo
        self.calls = []
        self.readback = None

    def get(self, path):
        self.calls.append(("GET", path))
        if path.endswith("/pulls/1"):
            return {
                "state": self.state,
                "head": {"sha": self.sha, "repo": {"full_name": self.head_repo}},
                "base": {"repo": {"full_name": REPOSITORY}},
            }
        if "/issues/1/comments?" in path:
            page = int(path.split("page=")[-1])
            return self.comments[(page - 1) * 100 : page * 100]
        identifier = int(path.rsplit("/", 1)[-1])
        if self.readback is not None:
            return self.readback
        return next(item for item in self.comments if item["id"] == identifier)

    def post(self, path, payload):
        self.calls.append(("POST", path, payload))
        created = comment(1000, payload["body"])
        self.comments.append(created)
        return created

    def patch(self, path, payload):
        self.calls.append(("PATCH", path, payload))
        identifier = int(path.rsplit("/", 1)[-1])
        original = next(item for item in self.comments if item["id"] == identifier)
        original["body"] = payload["body"]
        return original


def invoke(api, directory):
    return feedback.comment_report(api, REPOSITORY, 123, directory, BOT)


def test_creates_only_after_rechecking_head_and_verifies_body(tmp_path):
    prepare(tmp_path)
    api = FakeAPI()
    result = invoke(api, tmp_path)
    assert result["comment_action"] == "created"
    assert result["comment_url"].endswith("#issuecomment-1000")
    assert api.calls[-1] == ("GET", "/repos/owner/consumer/issues/comments/1000")
    post_index = next(index for index, call in enumerate(api.calls) if call[0] == "POST")
    assert api.calls[post_index - 1] == ("GET", "/repos/owner/consumer/pulls/1")
    body = api.comments[0]["body"]
    assert feedback.MARKER in body
    assert feedback.run_marker(123, 1) in body
    assert SHA in body
    assert "@somebody" not in body
    assert "@\u200bsomebody" in body


@pytest.mark.parametrize("options", [{"sha": "b" * 40}, {"state": "closed"}, {"head_repo": "attacker/fork"}])
def test_non_current_or_closed_or_fork_pr_is_never_mutated(tmp_path, options):
    prepare(tmp_path)
    api = FakeAPI(**options)
    with pytest.raises(feedback.FeedbackError):
        invoke(api, tmp_path)
    assert not any(call[0] in {"POST", "PATCH"} for call in api.calls)


def test_head_change_during_comment_listing_is_rejected(tmp_path):
    prepare(tmp_path)

    class ChangingAPI(FakeAPI):
        def get(self, path):
            response = super().get(path)
            if "/comments?" in path:
                self.sha = "b" * 40
            return response

    api = ChangingAPI()
    with pytest.raises(feedback.FeedbackError):
        invoke(api, tmp_path)
    assert not any(call[0] in {"POST", "PATCH"} for call in api.calls)


def test_human_marker_is_not_owned_and_is_never_overwritten(tmp_path):
    prepare(tmp_path)
    human = comment(5, feedback.MARKER, login="maintainer", user_type="User")
    api = FakeAPI(comments=[human])
    invoke(api, tmp_path)
    assert human["body"] == feedback.MARKER
    assert any(call[0] == "POST" for call in api.calls)
    assert not any(call[0] == "PATCH" for call in api.calls)


def test_updates_latest_own_bot_comment_and_is_idempotent(tmp_path):
    prepare(tmp_path)
    first = comment(1, feedback.MARKER + feedback.run_marker(122, 1))
    second = comment(2, feedback.MARKER + feedback.run_marker(122, 1))
    api = FakeAPI(comments=[first, second])
    result = invoke(api, tmp_path)
    assert result["comment_id"] == "2"
    assert any(call[0] == "PATCH" and call[1].endswith("/2") for call in api.calls)
    count = sum(call[0] in {"POST", "PATCH"} for call in api.calls)
    assert invoke(api, tmp_path)["comment_action"] == "unchanged"
    assert sum(call[0] in {"POST", "PATCH"} for call in api.calls) == count


def test_newer_report_on_any_owned_comment_blocks_stale_result(tmp_path):
    prepare(tmp_path)
    api = FakeAPI(
        comments=[
            comment(1, feedback.MARKER + feedback.run_marker(124, 1)),
            comment(2, feedback.MARKER + feedback.run_marker(122, 1)),
        ]
    )
    with pytest.raises(feedback.FeedbackError):
        invoke(api, tmp_path)
    assert not any(call[0] in {"POST", "PATCH"} for call in api.calls)


def test_later_attempt_blocks_earlier_attempt(tmp_path):
    prepare(tmp_path)
    api = FakeAPI(comments=[comment(1, feedback.MARKER + feedback.run_marker(123, 2))])
    with pytest.raises(feedback.FeedbackError):
        invoke(api, tmp_path)


def test_paginates_without_modifying_comments_on_other_pages(tmp_path):
    prepare(tmp_path)
    humans = [comment(index, "ordinary human text", login="maintainer", user_type="User") for index in range(1, 101)]
    api = FakeAPI(comments=[*humans, comment(101, feedback.MARKER + feedback.run_marker(122, 1))])
    result = invoke(api, tmp_path)
    assert result["comment_id"] == "101"
    assert any("page=2" in call[1] for call in api.calls)


def test_excessive_comment_pages_fail_closed(tmp_path):
    prepare(tmp_path)
    api = FakeAPI(comments=[comment(index, "ordinary") for index in range(1, 1001)])
    with pytest.raises(feedback.FeedbackError):
        invoke(api, tmp_path)
    assert not any(call[0] in {"POST", "PATCH"} for call in api.calls)


def test_readback_must_match_exact_body_and_author(tmp_path):
    prepare(tmp_path)
    api = FakeAPI()
    api.readback = comment(1000, "different body")
    with pytest.raises(feedback.FeedbackError):
        invoke(api, tmp_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"is_current": "false"},
        {"pr_number": "0"},
        {"source_head_sha": "bad"},
        {"run_url": "https://github.com/other/repo/actions/runs/123"},
        {"run_attempt": "0"},
    ],
)
def test_feedback_metadata_cannot_redirect_comment_target(tmp_path, changes):
    prepare(tmp_path, **changes)
    api = FakeAPI()
    with pytest.raises(feedback.FeedbackError):
        invoke(api, tmp_path)
    assert not api.calls


def test_oversized_comment_and_symlinked_feedback_are_rejected(tmp_path):
    prepare(tmp_path)
    (tmp_path / "pr-comment.md").write_text("x" * 60000)
    with pytest.raises(feedback.FeedbackError):
        invoke(FakeAPI(), tmp_path)
    (tmp_path / "pr-comment.md").unlink()
    (tmp_path / "pr-comment.md").symlink_to(tmp_path / "publication.json")
    with pytest.raises(feedback.FeedbackError):
        invoke(FakeAPI(), tmp_path)
