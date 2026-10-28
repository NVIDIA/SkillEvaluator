# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- standalone trusted workflow entrypoint with fixed diagnostics.

"""Upsert verified current-head PR feedback after a trusted Pages deployment.

Only the configured bot's marked comments are eligible for update. This helper
does not check out PR code, execute artifacts, or print the comment body.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

MARKER = "<!-- skillevaluator-tier1-report -->"
RUN_MARKER = re.compile(r"<!-- skillevaluator-tier1-run:([1-9][0-9]*):([1-9][0-9]*) -->")
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
BOT_PATTERN = re.compile(r"[A-Za-z0-9_-]+\[bot\]")
MAX_BODY_BYTES = 60000
MAX_METADATA_BYTES = 64 * 1024
MAX_API_BYTES = 16 * 1024 * 1024
MAX_PAGES = 10


class FeedbackError(ValueError):
    """A fixed diagnostic safe to print without feedback content or credentials."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _request, _file, _code, _message, _headers, _url):
        return None


class GitHubAPI:
    def __init__(self, repository: str, token: str):
        self.repository, self.token = repository, token
        self.opener = urllib.request.build_opener(_NoRedirect())

    def _request(self, method: str, path: str, payload=None):
        prefix = f"/repos/{self.repository}/"
        if not path.startswith(prefix) or ".." in path or "\\" in path:
            raise FeedbackError("Only this repository's GitHub API endpoints are supported.")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "SkillEvaluator-PR-feedback",
        }
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(f"https://api.github.com{path}", data=data, headers=headers, method=method)
        with self.opener.open(request, timeout=60) as response:
            raw = response.read(MAX_API_BYTES + 1)
        if len(raw) > MAX_API_BYTES:
            raise FeedbackError("The GitHub API response exceeded its size limit.")
        return _json(raw)

    def get(self, path: str):
        return self._request("GET", path)

    def post(self, path: str, payload: dict):
        return self._request("POST", path, payload)

    def patch(self, path: str, payload: dict):
        return self._request("PATCH", path, payload)


def _json(raw: bytes):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise FeedbackError("Feedback metadata contains duplicate JSON keys.")
            value[key] = item
        return value

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise FeedbackError("Feedback metadata or the GitHub API response is invalid JSON.") from exc


def _read(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise FeedbackError("A required feedback file is missing, linked, or oversized.")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise FeedbackError("A required feedback file exceeds its size limit.")
    return raw


def _positive(value) -> int:
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]{0,19}", value) is None:
        raise FeedbackError("A run, attempt, or pull request identity is invalid.")
    return int(value)


def run_marker(run_id: int, attempt: int) -> str:
    return f"<!-- skillevaluator-tier1-run:{run_id}:{attempt} -->"


def _verify_pr(api, repository: str, number: int, sha: str) -> str | None:
    pr = api.get(f"/repos/{repository}/pulls/{number}")
    if not isinstance(pr, dict) or pr.get("state") not in {"open", "closed"}:
        raise FeedbackError("The pull request lacks a valid state.")
    if pr["state"] == "closed":
        return "The pull request is closed."
    head = pr.get("head")
    base = pr.get("base")
    if not isinstance(head, dict) or not isinstance(base, dict):
        raise FeedbackError("The pull request lacks trustworthy repository and revision metadata.")
    if not isinstance(head.get("sha"), str) or SHA_PATTERN.fullmatch(head["sha"]) is None:
        raise FeedbackError("The pull request lacks a valid head revision.")
    if (head.get("repo") or {}).get("full_name") != repository or (base.get("repo") or {}).get(
        "full_name"
    ) != repository:
        raise FeedbackError("The pull request belongs to another repository; feedback was not posted.")
    if head.get("sha") != sha:
        return "The pull request head changed."
    return None


def _owned(comment: dict, bot: str) -> bool:
    user = comment.get("user")
    return (
        isinstance(user, dict)
        and user.get("login") == bot
        and user.get("type") == "Bot"
        and isinstance(comment.get("body"), str)
        and MARKER in comment["body"]
    )


def _owned_comments(api, repository: str, number: int, bot: str) -> list[dict]:
    owned = []
    for page in range(1, MAX_PAGES + 1):
        comments = api.get(f"/repos/{repository}/issues/{number}/comments?per_page=100&page={page}")
        if (
            not isinstance(comments, list)
            or len(comments) > 100
            or any(not isinstance(item, dict) for item in comments)
        ):
            raise FeedbackError("The pull request comment listing is invalid.")
        for item in comments:
            if _owned(item, bot):
                if type(item.get("id")) is not int or item["id"] <= 0:
                    raise FeedbackError("An existing bot comment has an invalid identity.")
                owned.append(item)
        if len(comments) < 100:
            return owned
    raise FeedbackError("The pull request comment listing exceeded the pagination limit.")


def _comment_identity(comment: dict) -> tuple[int, int]:
    identities = RUN_MARKER.findall(comment["body"])
    if len(identities) != 1:
        raise FeedbackError("An owned report comment lacks a unique run receipt; it was not overwritten.")
    return tuple(int(part) for part in identities[0])


def _verified_comment(api, repository: str, number: int, identifier: int, bot: str, body: str) -> str:
    comment = api.get(f"/repos/{repository}/issues/comments/{identifier}")
    if (
        not isinstance(comment, dict)
        or comment.get("id") != identifier
        or not _owned(comment, bot)
        or comment.get("body") != body
    ):
        raise FeedbackError("GitHub did not return the exact expected bot comment on readback.")
    expected = f"https://github.com/{repository}/pull/{number}#issuecomment-{identifier}"
    if comment.get("html_url") != expected:
        raise FeedbackError("The comment readback URL does not match the expected pull request.")
    return expected


def comment_report(api, repository: str, run_id: int, feedback_dir: Path, actor_bot_login: str) -> dict[str, str]:
    """Upsert only a current, non-stale, verified owned PR comment."""
    if not REPOSITORY_PATTERN.fullmatch(repository) or type(run_id) is not int or run_id <= 0:
        raise FeedbackError("An explicit repository and positive report run ID are required.")
    if not BOT_PATTERN.fullmatch(actor_bot_login):
        raise FeedbackError("ACTOR_BOT_LOGIN must name the bot permitted to own report comments.")
    if feedback_dir.is_symlink() or not feedback_dir.is_dir():
        raise FeedbackError("FEEDBACK_DIR must be a regular feedback directory.")
    metadata = _json(_read(feedback_dir / "publication.json", MAX_METADATA_BYTES))
    if not isinstance(metadata, dict):
        raise FeedbackError("The publication metadata is invalid.")
    expected_run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    if metadata.get("run_url") != expected_run_url or metadata.get("is_current") != "true":
        raise FeedbackError("The feedback is not a current report from the expected repository and run.")
    number = _positive(metadata.get("pr_number"))
    attempt = _positive(metadata.get("run_attempt"))
    sha = metadata.get("source_head_sha")
    if not isinstance(sha, str) or SHA_PATTERN.fullmatch(sha) is None:
        raise FeedbackError("The feedback source revision is invalid.")
    try:
        message = _read(feedback_dir / "pr-comment.md", MAX_BODY_BYTES).decode("utf-8")
    except UnicodeError as exc:
        raise FeedbackError("The feedback body is not valid UTF-8.") from exc
    if not message.strip() or MARKER in message or RUN_MARKER.search(message):
        raise FeedbackError("The rendered feedback is empty or contains reserved report markers.")
    # Defense in depth even when the trusted renderer already neutralized mentions.
    message = message.replace("@", "@\u200b")
    body = (
        f"{MARKER}\n{run_marker(run_id, attempt)}\n"
        f"Source commit: `{sha}` · [Evaluation run]({expected_run_url})\n\n{message}"
    )
    if len(body.encode("utf-8")) > MAX_BODY_BYTES:
        raise FeedbackError("The complete feedback comment exceeds the 60000-byte limit.")
    reason = _verify_pr(api, repository, number, sha)
    if reason:
        return {"comment_action": "skipped", "comment_reason": reason}
    owned = _owned_comments(api, repository, number, actor_bot_login)
    if any(_comment_identity(item) > (run_id, attempt) for item in owned):
        return {"comment_action": "skipped", "comment_reason": "A newer report comment already exists."}
    canonical = max(owned, key=lambda item: item["id"]) if owned else None
    # Recheck immediately before mutation after pagination and receipt validation.
    reason = _verify_pr(api, repository, number, sha)
    if reason:
        return {"comment_action": "skipped", "comment_reason": reason}
    if canonical is not None and canonical["body"] == body:
        identifier = canonical["id"]
        action = "unchanged"
    elif canonical is not None:
        identifier = canonical["id"]
        response = api.patch(f"/repos/{repository}/issues/comments/{identifier}", {"body": body})
        if not isinstance(response, dict) or response.get("id") != identifier:
            raise FeedbackError("GitHub did not acknowledge the expected comment update.")
        action = "updated"
    else:
        response = api.post(f"/repos/{repository}/issues/{number}/comments", {"body": body})
        if not isinstance(response, dict) or type(response.get("id")) is not int or response["id"] <= 0:
            raise FeedbackError("GitHub did not acknowledge a valid new comment identity.")
        identifier = response["id"]
        action = "created"
    url = _verified_comment(api, repository, number, identifier, actor_bot_login, body)
    return {"comment_url": url, "comment_id": str(identifier), "comment_action": action}


def main() -> int:
    try:
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        run_id = _positive(os.environ.get("REPORT_RUN_ID"))
        token = os.environ.get("GH_TOKEN", "").strip()
        if not token:
            raise FeedbackError("GH_TOKEN is required to publish the authorized PR feedback.")
        result = comment_report(
            GitHubAPI(repository, token),
            repository,
            run_id,
            Path(os.environ["FEEDBACK_DIR"]),
            os.environ.get("ACTOR_BOT_LOGIN", "github-actions[bot]"),
        )
        if os.environ.get("GITHUB_OUTPUT"):
            with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as handle:
                handle.write("".join(f"{key}={value}\n" for key, value in result.items()))
        if result["comment_action"] == "skipped":
            print("SkillEvaluator PR feedback skipped: " + result["comment_reason"])
        else:
            print("SkillEvaluator PR feedback verified successfully.")
        return 0
    except FeedbackError as exc:
        print("SkillEvaluator PR feedback rejected: " + json.dumps(str(exc)[:1000]), file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError, TypeError, RecursionError):
        print("SkillEvaluator PR feedback failed; inspect the trusted workflow configuration.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
