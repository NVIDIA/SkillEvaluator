# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- pinned trusted CI entrypoint with fixed diagnostics.
"""Propose eligible native benchmark cards through a dedicated GitHub PR."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.parse
from pathlib import Path

if __package__:
    from . import feedback, github_api, publish_reports
    from .benchmark_reports import benchmark_documents
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import feedback
    import github_api
    import publish_reports
    from benchmark_reports import benchmark_documents

BRANCH = "skillevaluator/benchmark-update"
MARKER = "<!-- skillevaluator-benchmark-update -->"
PR_DISABLED = "Benchmark cards are ready on the dedicated bot branch. Enable GitHub Actions permission to create pull requests, or open a pull request from skillevaluator/benchmark-update manually."
_require = feedback._require


def _sha(value):
    _require(isinstance(value, str) and github_api.COMMIT_SHA.fullmatch(value), "A Git object identity is invalid.")
    return value


def _optional(api, path):
    try:
        return api.get(path)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        return None


def _head(api, root, branch):
    value = api.get(f"{root}/git/ref/heads/{urllib.parse.quote(branch, safe='/')}")
    _require(value.get("object", {}).get("type") == "commit", "A branch does not identify a commit.")
    return _sha(value["object"].get("sha"))


def _tree(api, root, commit):
    value = api.get(f"{root}/git/commits/{commit}")
    sha = _sha(value.get("tree", {}).get("sha"))
    result = api.get(f"{root}/git/trees/{sha}?recursive=1")
    _require(
        result.get("truncated") is False and isinstance(result.get("tree"), list), "The repository tree is incomplete."
    )
    entries = {}
    for entry in result["tree"]:
        path = entry.get("path")
        # Unrelated repository paths are only compared, never written; written
        # card and receipt paths get their own _safe_path checks.
        _require(
            isinstance(path, str) and 0 < len(path) <= 4096 and path not in entries,
            "The repository tree has an invalid path.",
        )
        _sha(entry.get("sha"))
        entries[path] = entry
    return sha, entries


def _regular_targets(entries, targets):
    for path in targets:
        parts = path.split("/")
        for index in range(1, len(parts)):
            ancestor = entries.get("/".join(parts[:index]))
            _require(
                ancestor is not None and ancestor.get("mode") == "040000" and ancestor.get("type") == "tree",
                "A benchmark parent is not a regular source directory.",
            )
        entry = entries.get(path)
        _require(
            entry is None or (entry.get("mode") in {"100644", "100755"} and entry.get("type") == "blob"),
            "An existing benchmark target is not a regular file.",
        )


def _pull_identity(pull):
    base, user = pull.get("base") or {}, pull.get("user") or {}
    return (
        pull.get("number"),
        pull.get("state"),
        pull.get("title"),
        pull.get("body"),
        base.get("ref"),
        (base.get("repo") or {}).get("full_name"),
        user.get("login"),
        user.get("type"),
    )


def _blob(content):
    payload = content.encode("utf-8")
    return hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload, usedforsecurity=False
    ).hexdigest()


def github_benchmark_publish(api, repository: str, run_id: int, attempt: int) -> dict:
    """Authenticate evidence, regenerate cards, then propose only eligible cards.

    No source checkout or artifact Markdown is read. Updates fast-forward the
    dedicated branch; unrelated branch differences require manual reconciliation.
    """
    native = {}
    evidence = feedback.verify_github_evidence(
        api, repository, run_id, attempt, events=frozenset({"push", "workflow_dispatch"}), native_reports=native
    )
    run = evidence["run"]
    source = run["head_sha"]
    root = f"/repos/{repository}"
    info = api.get(root)
    default = info.get("default_branch")
    _require(
        isinstance(default, str)
        and feedback._safe_path(default)
        and default != BRANCH
        and run.get("head_branch") == default,
        "The producer must target the current default branch.",
    )

    def current():
        latest = api.get(f"{root}/actions/runs/{run_id}")
        return (
            _head(api, root, default) == source
            and latest.get("id") == run_id
            and latest.get("run_attempt") == attempt
            and latest.get("status") == "completed"
            and latest.get("head_sha") == source
            and latest.get("head_branch") == default
            and latest.get("event") == run["event"]
            and latest.get("repository", {}).get("full_name") == repository
            and latest.get("head_repository", {}).get("full_name") == repository
        )

    if not current():
        # A newer default-branch push or attempt supersedes this producer; that
        # is routine on an active branch, not a publication failure.
        return {
            "benchmark_action": "skipped",
            "benchmark_reason": "The source head, producer attempt or bot branch changed.",
        }
    targets = {}
    for document in benchmark_documents(evidence["manifest"], native):
        if not document["eligible"]:
            continue
        skill = document["path"]
        _require(skill == "." or feedback._safe_path(skill), "A benchmark source path is invalid.")
        path = "BENCHMARK.md" if skill == "." else skill + "/BENCHMARK.md"
        _require(
            path not in targets and isinstance(document["benchmark_content"], str), "A benchmark target is ambiguous."
        )
        targets[path] = document["benchmark_content"]
    if not targets:
        return {"benchmark_action": "unchanged", "benchmark_reason": "No eligible benchmark cards."}
    source_tree, source_entries = _tree(api, root, source)
    _regular_targets(source_entries, targets)
    # A card already identical on the default branch needs no review branch.
    if all(source_entries.get(path, {}).get("sha") == _blob(content) for path, content in targets.items()):
        return {"benchmark_action": "unchanged"}
    branch_ref = _optional(api, f"{root}/git/ref/heads/{BRANCH}")
    branch = None
    base_tree, entries = source_tree, source_entries
    if branch_ref is not None:
        _require(branch_ref.get("object", {}).get("type") == "commit", "The bot branch is not a commit.")
        branch = _sha(branch_ref["object"].get("sha"))
        previous = api.get(f"{root}/git/commits/{branch}")
        message = previous.get("message")
        _require(
            isinstance(message, str)
            and len(message.encode("utf-8")) <= 1024 * 1024
            and message.startswith("Update eligible SkillEvaluator benchmark cards\n\n"),
            "The bot branch lacks its publication source receipt.",
        )
        trailers = [line for line in message.splitlines() if line.startswith("Source:")]
        _require(
            len(trailers) == 1 and re.fullmatch(r"Source: [0-9a-f]{40}", trailers[0]),
            "The bot branch source trailer is missing or ambiguous.",
        )
        previous_source = trailers[0].removeprefix("Source: ")
        if previous_source != source:
            relation = api.get(f"{root}/compare/{previous_source}...{source}")
            _require(
                relation.get("status") == "ahead"
                and relation.get("merge_base_commit", {}).get("sha") == previous_source,
                "The recorded publication source is not in current default-branch history.",
            )
        _, prior_entries = _tree(api, root, previous_source)
        _, branch_entries = _tree(api, root, branch)
        # Validate the old proposal against its own immutable source, never
        # against the newer default tree. No old branch files are copied.
        receipts = [line for line in message.splitlines() if line.startswith("Benchmark-Cards:")]
        _require(
            len(receipts) == 1 and receipts[0].startswith("Benchmark-Cards: "),
            "The bot branch card receipt is missing or ambiguous.",
        )
        cards = github_api.strict_json(
            receipts[0].removeprefix("Benchmark-Cards: ").encode("utf-8"),
            feedback.FeedbackError,
            "The bot branch card receipt is invalid.",
        )
        _require(
            isinstance(cards, dict) and 0 < len(cards) <= feedback.MAX_SKILLS, "The bot branch card receipt is invalid."
        )
        for path, blob in cards.items():
            _require(
                feedback._safe_path(path) and path.rsplit("/", maxsplit=1)[-1] == "BENCHMARK.md",
                "The bot branch card receipt has an unsafe path.",
            )
            _sha(blob)
            entry = branch_entries.get(path, {})
            _require(
                entry.get("sha") == blob and entry.get("mode") == "100644" and entry.get("type") == "blob",
                "A previously generated card was changed or removed.",
            )
        _regular_targets(prior_entries, cards)
        _regular_targets(branch_entries, cards)
        ancestors = {"/".join(path.split("/")[:index]) for path in cards for index in range(1, len(path.split("/")))}

        def unrelated(rows):
            return {
                path: (row.get("mode"), row.get("type"), row.get("sha"))
                for path, row in rows.items()
                if path not in cards and path not in ancestors
            }

        _require(
            unrelated(branch_entries) == unrelated(prior_entries), "The bot branch contains unrelated source changes."
        )
        # Reuse a matching proposal on the same source; otherwise build the
        # complete desired tree from current source plus newly generated cards.
        desired_matches_branch = (
            previous_source == source
            and all(branch_entries.get(path, {}).get("sha") == _blob(content) for path, content in targets.items())
            and all(path in targets or branch_entries.get(path) == source_entries.get(path) for path in cards)
        )
        if desired_matches_branch:
            entries = branch_entries
    pulls = feedback._listing(
        api,
        f"{root}/pulls?state=open&head={urllib.parse.quote(repository.split('/', maxsplit=1)[0] + ':' + BRANCH, safe='')}",
    )
    _require(len(pulls) <= 1, "Multiple benchmark pull requests require reconciliation.")
    pull = pulls[0] if pulls else None
    if pull:
        _require(
            pull.get("state") == "open"
            and pull.get("head", {}).get("ref") == BRANCH
            and pull.get("base", {}).get("ref") == default
            and all(pull.get(part, {}).get("repo", {}).get("full_name") == repository for part in ("head", "base"))
            and pull.get("user", {}).get("type") == "Bot"
            and pull["user"].get("login") == "github-actions[bot]"
            and isinstance(pull.get("body"), str)
            and pull["body"].startswith(MARKER + "\n"),
            "The existing benchmark PR is not owned by this publisher.",
        )
        feedback._positive(pull.get("number"))
    changes = [
        {"path": path, "mode": "100644", "type": "blob", "content": content}
        for path, content in sorted(targets.items())
        if entries.get(path, {}).get("sha") != _blob(content)
    ]

    def guarded():
        if not current():
            return False
        # Ref rereads protect against concurrent human edits; force is never used.
        latest = _optional(api, f"{root}/git/ref/heads/{BRANCH}")
        return (
            latest is None if branch is None else latest is not None and latest.get("object", {}).get("sha") == branch
        )

    if not guarded():
        return {
            "benchmark_action": "skipped",
            "benchmark_reason": "The source head, producer attempt or bot branch changed.",
        }
    if changes:
        commit_message = (
            f"Update eligible SkillEvaluator benchmark cards\n\nSource: {source}\nRun: {run_id}/{attempt}\nBenchmark-Cards: "
            + json.dumps(
                {path: _blob(content) for path, content in targets.items()},
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
        _require(
            len(commit_message.encode("utf-8")) <= 1024 * 1024, "The benchmark commit receipt exceeds its size limit."
        )
        tree = api.post(f"{root}/git/trees", {"base_tree": base_tree, "tree": changes})
        if not guarded():
            return {
                "benchmark_action": "skipped",
                "benchmark_reason": "The source head, producer attempt or bot branch changed.",
            }
        commit = api.post(
            f"{root}/git/commits",
            {
                "message": commit_message,
                "tree": _sha(tree.get("sha")),
                "parents": list(dict.fromkeys([branch, source] if branch else [source])),
            },
        )
        if not guarded():
            return {
                "benchmark_action": "skipped",
                "benchmark_reason": "The source head, producer attempt or bot branch changed.",
            }
        commit_sha = _sha(commit.get("sha"))
        if branch is None:
            response = api.post(f"{root}/git/refs", {"ref": f"refs/heads/{BRANCH}", "sha": commit_sha})
        else:
            response = api.patch(f"{root}/git/refs/heads/{BRANCH}", {"sha": commit_sha, "force": False})
        _require(response.get("object", {}).get("sha") == commit_sha, "The bot branch update was not acknowledged.")
        branch = commit_sha
    if not guarded():
        return {
            "benchmark_action": "skipped",
            "benchmark_reason": "The source head, producer attempt or bot branch changed.",
        }
    report = f"https://github.com/{repository}/actions/runs/{run_id}/artifacts/{evidence['artifact']['id']}"
    body = f"{MARKER}\nRegenerated eligible benchmark cards from validated canonical native JSON.\n\nSource commit: `{source}`\n\n[Evaluation reports]({report}) · attempt {attempt}\n"
    payload = {"title": "Update SkillEvaluator benchmark cards", "body": body}
    if pull:
        if pull.get("body") == body and pull.get("title") == payload["title"]:
            return {
                "benchmark_action": "updated" if changes else "unchanged",
                "benchmark_pr_url": f"https://github.com/{repository}/pull/{pull['number']}",
            }
        check = api.get(f"{root}/pulls/{pull['number']}")
        # base.repo is the live repository object (pushed_at, counters), and the
        # branch update above is itself a push. Compare only stable PR identity.
        _require(_pull_identity(check) == _pull_identity(pull), "The benchmark PR changed before update.")
        _require(
            check.get("head", {}).get("ref") == BRANCH
            and check.get("head", {}).get("repo", {}).get("full_name") == repository
            and check.get("head", {}).get("sha") == branch,
            "The benchmark PR head changed before update.",
        )
        if not guarded():
            return {
                "benchmark_action": "skipped",
                "benchmark_reason": "The source head, producer attempt or bot branch changed.",
            }
        response = api.patch(f"{root}/pulls/{pull['number']}", payload)
        action = "updated"
    else:
        try:
            response = api.post(f"{root}/pulls", {**payload, "base": default, "head": BRANCH})
        except urllib.error.HTTPError as error:
            if error.code != 403:
                raise
            return {"benchmark_action": "branch_ready", "benchmark_reason": PR_DISABLED}
        action = "created"
    number = feedback._positive(response.get("number"))
    return {"benchmark_action": action, "benchmark_pr_url": f"https://github.com/{repository}/pull/{number}"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--run-attempt", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        _require(bool(token), "GITHUB_TOKEN is required.")
        result = github_benchmark_publish(
            publish_reports.GitHubAPI(args.repository, token), args.repository, args.run_id, args.run_attempt
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (
        feedback.FeedbackError,
        publish_reports.PublicationError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        urllib.error.URLError,
        http.client.HTTPException,
    ):
        print(
            "Trusted benchmark publication could not validate its evidence, current default head, or dedicated bot branch.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
