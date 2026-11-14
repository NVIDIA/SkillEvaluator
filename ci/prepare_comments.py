# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- standalone trusted workflow entrypoint with fixed diagnostics.

"""Render and post native PR feedback from verified scan JSON, without hosting."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
from pathlib import Path

try:
    import comment_report
    import publish_reports
    import report_views
except ModuleNotFoundError:
    from ci import comment_report, publish_reports, report_views


def prepare_comments(api, repository: str, run_id: int, run_attempt: int, output: Path) -> dict[str, str]:
    """Reuse the publisher's evidence checks and the native Markdown/comment path."""
    if (
        not publish_reports.REPO_PATTERN.fullmatch(repository)
        or not publish_reports._integer(run_id, 1)
        or not publish_reports._integer(run_attempt, 1)
    ):
        raise publish_reports.PublicationError("Repository, run ID, and attempt must be explicit valid identities.")
    reason = comment_report._verify_run(api, repository, run_id, run_attempt)
    if reason:
        return {"comment_action": "skipped", "comment_reason": reason}
    evidence = publish_reports.load_report_evidence(api, repository, run_id, run_attempt)
    run, summary = evidence["run"], evidence["summary"]
    if run["event"] != "pull_request":
        return {"comment_action": "skipped", "comment_reason": "The scan is not a pull request run."}
    routes, number = publish_reports._current_routes(api, repository, run)
    if not number or routes != [f"pr/{number}"]:
        return {"comment_action": "skipped", "comment_reason": "No unique current same-repository pull request."}
    if output.is_symlink():
        raise publish_reports.PublicationError("The feedback directory must not be a symlink.")
    output.mkdir(parents=True, exist_ok=True)
    publish_reports._dump(output / "summary.json", summary)
    for directory, report in evidence["reports"].items():
        publish_reports._dump(output / directory / "skillevaluator-tier1.json", report)
    run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    report_views.render_reports(
        output,
        summary,
        {
            "run_url": run_url,
            "commit": run["head_sha"],
            "image": evidence["image"],
            "llm_enabled": evidence["llm_enabled"],
            "ci_conclusion": run["conclusion"],
        },
    )
    publish_reports._dump(
        output / "publication.json",
        {
            "run_url": run_url,
            "pr_number": number,
            "source_head_sha": run["head_sha"],
            "is_current": "true",
            "run_attempt": str(run_attempt),
        },
    )
    return comment_report.comment_report(api, repository, run_id, output, "github-actions[bot]")


def main() -> int:
    try:
        token = os.environ.get("GH_TOKEN", "").strip()
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        run_id = comment_report._positive(os.environ.get("REPORT_RUN_ID"))
        attempt = comment_report._positive(os.environ.get("REPORT_RUN_ATTEMPT"))
        if not token:
            raise publish_reports.PublicationError("GH_TOKEN is required to publish authorized PR feedback.")
        with tempfile.TemporaryDirectory(prefix="skillevaluator-feedback-") as directory:
            output = Path(directory)
            result = prepare_comments(publish_reports.GitHubAPI(repository, token), repository, run_id, attempt, output)
            if os.environ.get("GITHUB_STEP_SUMMARY") and (output / "summary.md").is_file():
                with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as handle:
                    handle.write((output / "summary.md").read_text(encoding="utf-8"))
        if result["comment_action"] == "skipped":
            print("SkillEvaluator PR feedback skipped: " + result["comment_reason"])
        else:
            print("SkillEvaluator PR feedback verified successfully.")
        return 0
    except publish_reports.StalePublication as exc:
        print("SkillEvaluator PR feedback skipped: " + str(exc))
        return 0
    except (publish_reports.PublicationError, comment_report.FeedbackError) as exc:
        print("SkillEvaluator PR feedback rejected: " + json.dumps(str(exc)[:1000]), file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError, TypeError, RecursionError, urllib.error.URLError):
        print("SkillEvaluator PR feedback failed; inspect the trusted workflow configuration.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
