# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- standalone trusted publisher; stdout is fixed.

"""Publish bounded report data without executing checkout or artifact code.

Run only from a trusted revision in a privileged workflow. This helper writes
static files; the workflow owns Git commits, Pages deployment and PR comments.
TRUSTED_EVALUATOR_SHA and TRUSTED_CALLER_BLOBS must explicitly allow the reusable
workflow commit and exact caller workflow blob; neither has a permissive default.
"""

from __future__ import annotations

import html
import io
import json
import os
import re
import shutil
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from skillevaluator.utils.secret_redaction import REDACTED, SENSITIVE_PII_CATEGORIES, Redactor, sensitive_values

MAX_FILES = 1000
MAX_SKILLS = 100
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_HISTORY_BYTES = 512 * 1024 * 1024
ARTIFACT_NAME = "skill-evaluator-tier1"
WORKFLOW_PATH = ".github/workflows/skill-checks.yml"
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
REPO_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
REPORT_DIRECTORY = re.compile(r"skill-(?:00[1-9]|0[1-9][0-9]|100)")


class PublicationError(ValueError):
    """Publication failed without exposing tokens or untrusted diagnostics."""


class StalePublication(PublicationError):
    """A PR has closed or moved since the selected run completed."""


def allowed_download_url(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    return (
        parsed.scheme == "https"
        and not parsed.username
        and not parsed.password
        and parsed.port in {None, 443}
        and any(
            host.endswith(suffix) and host != suffix[1:]
            for suffix in (".blob.core.windows.net", ".githubusercontent.com", ".actions.githubusercontent.com")
        )
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _request, _file, _code, _message, _headers, _url):
        return None


def _bounded_read(response, limit: int) -> bytes:
    declared = response.headers.get("Content-Length")
    if declared and (not declared.isdigit() or int(declared) > limit):
        raise PublicationError("The downloaded artifact or API response exceeds its size limit.")
    payload = response.read(limit + 1)
    if len(payload) > limit:
        raise PublicationError("The downloaded artifact or API response exceeds its size limit.")
    return payload


class GitHubAPI:
    def __init__(self, repository: str, token: str):
        self.repository = repository
        self.token = token
        self.opener = urllib.request.build_opener(_NoRedirect())

    def _request(self, path: str):
        root = f"/repos/{self.repository}"
        prefix = root + "/"
        comparison = re.fullmatch(re.escape(prefix) + r"compare/[0-9a-f]{40}\.\.\.[0-9a-f]{40}", path)
        if (path != root and not path.startswith(prefix)) or (".." in path and not comparison) or "\\" in path:
            raise PublicationError("Only this repository's GitHub API endpoints are supported.")
        request = urllib.request.Request(
            f"https://api.github.com{path}",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "SkillEvaluator-report-publisher",
            },
        )
        return self.opener.open(request, timeout=60)

    def get(self, path: str):
        with self._request(path) as response:
            return _json(_bounded_read(response, MAX_FILE_BYTES))

    def download_artifact(self, artifact_id: int) -> bytes:
        try:
            with self._request(f"/repos/{self.repository}/actions/artifacts/{artifact_id}/zip") as response:
                return _bounded_read(response, MAX_TOTAL_BYTES)
        except urllib.error.HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                raise
            url = exc.headers.get("Location", "")
        # Never forward Authorization from api.github.com to a signed blob URL.
        for _ in range(5):
            if not allowed_download_url(url):
                raise PublicationError("The artifact download redirected to an unsupported host.")
            request = urllib.request.Request(url, headers={"User-Agent": "SkillEvaluator-report-publisher"})
            try:
                with self.opener.open(request, timeout=60) as response:
                    return _bounded_read(response, MAX_TOTAL_BYTES)
            except urllib.error.HTTPError as exc:
                if exc.code not in {301, 302, 303, 307, 308}:
                    raise
                url = urllib.parse.urljoin(url, exc.headers.get("Location", ""))
        raise PublicationError("The artifact download exceeded the redirect limit.")


def _json(payload: bytes):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PublicationError("Duplicate JSON keys are not supported.")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise PublicationError("Non-finite JSON numbers are not supported.")

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=object_pairs, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise PublicationError("A required JSON document is invalid.") from exc


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _scanner_list(value):
    return isinstance(value, list) and len(value) <= MAX_FILES and all(isinstance(item, str) for item in value)


def validate_summary(summary: dict) -> None:
    if not isinstance(summary, dict) or not isinstance(summary.get("skills"), list):
        raise PublicationError("The report summary lacks its skill list.")
    rows = summary["skills"]
    if len(rows) > MAX_SKILLS or any(not isinstance(row, dict) for row in rows):
        raise PublicationError("The summary has an unsupported skill list.")
    if any(not _integer(summary.get(key)) for key in ("total", "passed", "failed", "incomplete")):
        raise PublicationError("The summary contains invalid counts.")
    if type(summary.get("overall_passed")) is not bool or summary.get("overall_status") not in {
        "passed",
        "failed",
        "incomplete",
        "error",
        "skipped",
    }:
        raise PublicationError("The summary verdict is invalid.")
    directories = set()
    paths = set()
    for row in rows:
        directory, path = row.get("report_dir"), row.get("path")
        if not isinstance(directory, str) or REPORT_DIRECTORY.fullmatch(directory) is None or directory in directories:
            raise PublicationError("The summary contains an invalid report directory.")
        if not isinstance(path, str) or not path or path in paths:
            raise PublicationError("The summary contains invalid skill paths.")
        directories.add(directory)
        paths.add(path)
        if type(row.get("passed")) is not bool or row.get("overall_status") not in {
            "passed",
            "failed",
            "incomplete",
            "error",
        }:
            raise PublicationError("The summary contains an invalid skill verdict.")
        if not _scanner_list(row.get("incomplete_scans")):
            raise PublicationError("The summary lacks scanner completeness evidence.")
        if "exit_code" not in row or (row["exit_code"] is not None and type(row["exit_code"]) is not int):
            raise PublicationError("The summary contains an invalid process exit code.")
        if row["passed"] and (row["exit_code"] != 0 or row["overall_status"] != "passed" or row["incomplete_scans"]):
            raise PublicationError("A skill's verdict contradicts its process evidence.")
        if not row["passed"] and row["overall_status"] == "passed":
            raise PublicationError("A failed skill cannot claim a passing status.")
        if bool(row["incomplete_scans"]) != (row["overall_status"] == "incomplete"):
            raise PublicationError("A skill's status contradicts its scanner completeness evidence.")
        if row["overall_status"] != "error":
            if any(not _integer(row.get(key)) for key in ("total_validators", "total_errors", "total_warnings")):
                raise PublicationError("A skill summary lacks valid result counts.")
            counts = row.get("severity_counts")
            if not isinstance(counts, dict) or any(
                not _integer(counts.get(key)) for key in ("critical", "high", "medium", "low")
            ):
                raise PublicationError("A skill summary lacks valid severity counts.")
    passed = sum(row["passed"] for row in rows)
    incomplete = sum(row["overall_status"] in {"incomplete", "error"} for row in rows)
    if (summary["total"], summary["passed"], summary["failed"], summary["incomplete"]) != (
        len(rows),
        passed,
        len(rows) - passed,
        incomplete,
    ):
        raise PublicationError("The summary counts contradict its skill results.")
    expected = bool(rows) and passed == len(rows)
    if summary["overall_passed"] != expected or (summary["overall_status"] == "passed") != expected:
        raise PublicationError("The summary verdict contradicts its skill results.")
    skipped = summary["overall_status"] == "skipped"
    if skipped:
        selection = summary.get("selection")
        if (
            rows
            or summary.get("error")
            or not isinstance(selection, dict)
            or selection.get("mode") != "changed"
            or selection.get("selected") != []
            or not _integer(selection.get("discovered"))
            or not _integer(selection.get("changed_files"))
            or not isinstance(selection.get("deleted"), list)
            or any(not isinstance(path, str) for path in selection["deleted"])
            or any(
                not isinstance(selection.get(key), str) or SHA_PATTERN.fullmatch(selection[key]) is None
                for key in ("base", "head")
            )
        ):
            raise PublicationError("A no-change result requires complete changed-skill selection evidence.")
    expected_status = (
        "skipped"
        if skipped
        else "passed"
        if expected
        else "incomplete"
        if incomplete
        else "failed"
        if rows
        else "error"
    )
    if summary["overall_status"] != expected_status:
        raise PublicationError("The summary status contradicts its skill results.")
    for key in ("total_validators", "total_errors", "total_warnings"):
        if not _integer(summary.get(key)) or summary[key] != sum(row.get(key, 0) for row in rows):
            raise PublicationError("The summary aggregates contradict its per-skill result counts.")
    expected_severity = {
        key: sum(row.get("severity_counts", {}).get(key, 0) for row in rows)
        for key in ("critical", "high", "medium", "low")
    }
    if (
        not isinstance(summary.get("severity_counts"), dict)
        or any(not _integer(summary["severity_counts"].get(key)) for key in expected_severity)
        or summary["severity_counts"] != expected_severity
    ):
        raise PublicationError("The summary severity totals contradict its per-skill counts.")


def validate_report(report: dict, row: dict) -> None:
    if not isinstance(report, dict) or type(report.get("overall_passed")) is not bool:
        raise PublicationError("A per-skill report lacks a boolean verdict.")
    results = report.get("results")
    if not isinstance(results, list) or not results or any(not isinstance(result, dict) for result in results):
        raise PublicationError("A per-skill report lacks validator evidence.")
    if not _integer(report.get("total_validators"), 1) or report["total_validators"] != len(results):
        raise PublicationError("A per-skill report contains inconsistent validator counts.")
    if report.get("overall_status") not in {"passed", "failed", "incomplete"} or not _scanner_list(
        report.get("incomplete_scans")
    ):
        raise PublicationError("A per-skill report contains invalid completeness evidence.")
    incomplete = []
    totals = {"errors": 0, "warnings": 0, "critical_count": 0, "high_count": 0, "medium_count": 0, "low_count": 0}
    for result in results:
        status = result.get("status")
        if type(result.get("passed")) is not bool or status not in {"passed", "failed", "incomplete", "skipped"}:
            raise PublicationError("A validator has an invalid verdict.")
        scans = result.get("incomplete_scans")
        if not _scanner_list(scans) or (status == "incomplete") != bool(scans):
            raise PublicationError("A validator has contradictory scanner completeness evidence.")
        if result["passed"] != (status in {"passed", "skipped"}):
            raise PublicationError("A validator has contradictory pass and status fields.")
        incomplete.extend(scans)
        counts = result.get("summary")
        if not isinstance(counts, dict) or any(not _integer(counts.get(key)) for key in totals):
            raise PublicationError("A validator lacks trustworthy result counts.")
        for key in totals:
            totals[key] += counts[key]
    incomplete = list(dict.fromkeys(incomplete))
    all_passed = all(result["passed"] for result in results)
    status = "incomplete" if incomplete else "passed" if all_passed else "failed"
    if (report["overall_passed"], report["overall_status"], report["incomplete_scans"]) != (
        all_passed,
        status,
        incomplete,
    ):
        raise PublicationError("A report verdict contradicts its validator evidence.")
    if (row["passed"], row["overall_status"], row["incomplete_scans"]) != (all_passed, status, incomplete):
        raise PublicationError("A report verdict contradicts its per-skill summary.")
    if type(row["exit_code"]) is not int or (row["exit_code"] == 0) != all_passed:
        raise PublicationError("A report verdict contradicts its process exit code.")
    for field, key in (("total_errors", "errors"), ("total_warnings", "warnings")):
        if not _integer(report.get(field)) or report[field] != totals[key]:
            raise PublicationError("A report's error or warning counts contradict its validators.")
        if field in row and row[field] != report[field]:
            raise PublicationError("A skill summary has contradictory result counts.")
    expected_severity = {name: totals[f"{name}_count"] for name in ("critical", "high", "medium", "low")}
    if (
        not isinstance(report.get("severity_counts"), dict)
        or any(not _integer(report["severity_counts"].get(key)) for key in expected_severity)
        or report["severity_counts"] != expected_severity
    ):
        raise PublicationError("A report's severity counts contradict its validators.")
    if "severity_counts" in row and row["severity_counts"] != expected_severity:
        raise PublicationError("A skill summary has contradictory severity counts.")
    if "total_validators" in row and row["total_validators"] != report["total_validators"]:
        raise PublicationError("A skill summary has contradictory validator counts.")
    # Validate the same typed evidence the renderer consumes before writing an
    # immutable receipt. Valid-looking rollups alone cannot establish a report.
    from skillevaluator.reporting.retained import RetainedReportError, retained_results

    try:
        retained_results(report)
    except RetainedReportError as exc:
        raise PublicationError("A per-skill report contains invalid retained evidence.") from exc


def read_artifact(payload: bytes) -> tuple[dict, dict[str, dict]]:
    """Validate every member, but deserialize only expected JSON without extraction."""
    if len(payload) > MAX_TOTAL_BYTES:
        raise PublicationError("The artifact exceeds its download limit.")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_FILES:
                raise PublicationError("The artifact contains too many files.")
            names = set()
            total = 0
            expected = {}
            for entry in entries:
                name = entry.filename
                path = PurePosixPath(name)
                mode = entry.external_attr >> 16
                if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or name in names:
                    raise PublicationError("The artifact contains an unsafe or duplicate path.")
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                    raise PublicationError("The artifact contains linked or special files.")
                names.add(name)
                total += entry.file_size
                if entry.file_size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                    raise PublicationError("The artifact exceeds its expanded size limit.")
                if name == "summary.json" or re.fullmatch(r"skill-[0-9]{3}/skillevaluator-tier1\.json", name):
                    with archive.open(entry) as handle:
                        content = handle.read(MAX_FILE_BYTES + 1)
                    if len(content) > MAX_FILE_BYTES:
                        raise PublicationError("A report exceeds its size limit.")
                    expected[name] = content
    except (zipfile.BadZipFile, RuntimeError, OSError, ValueError) as exc:
        raise PublicationError("The report artifact archive is invalid.") from exc
    summary = _json(expected["summary.json"]) if "summary.json" in expected else None
    validate_summary(summary)
    reports = {}
    for row in summary["skills"]:
        directory = row["report_dir"]
        # A rejected/contradictory child report is diagnostic data, never evidence
        # to republish. The trusted renderer creates the unavailable-report page.
        if row["overall_status"] == "error":
            continue
        document = expected.get(f"{directory}/skillevaluator-tier1.json")
        if document is None:
            raise PublicationError("A required per-skill JSON report is missing.")
        document = _json(document)
        validate_report(document, row)
        reports[directory] = document
    # Historical trusted runs can predate source-level redaction. Collect raw
    # PII matches from their bounded JSON, then remove those values from every
    # retained object before creating history, HTML or PR feedback.
    tokens: set[str] = set()
    for report in reports.values():
        for result in report.get("results", []):
            for finding in result.get("findings", []):
                if not isinstance(finding, dict) or finding.get("category") != "PII":
                    continue
                category = finding.get("check_name")
                if category not in SENSITIVE_PII_CATEGORIES:
                    continue
                metadata = finding.get("metadata") or {}
                matched = metadata.get("matched_value") if isinstance(metadata, dict) else None
                line = finding.get("line_content")
                if isinstance(matched, str) and matched != REDACTED:
                    values = sensitive_values(category, matched, line if isinstance(line, str) else "")
                    # A PEM header does not identify its body, and an incomplete
                    # assignment can hide the value in another scanner's data.
                    if category == "private_keys" or (
                        category in {"hardcoded_secrets", "database_credentials"} and values == {matched}
                    ):
                        raise PublicationError("The artifact could not be safely redacted.")
                    tokens.update(values)
                elif isinstance(line, str) and line:
                    tokens.update(sensitive_values(category, line, line))
                elif matched != REDACTED:
                    raise PublicationError("The artifact could not be safely redacted.")
    try:
        redactor = Redactor(tokens)
        summary = redactor.object(summary)
        reports = redactor.object(reports)
        validate_summary(summary)
        for row in summary["skills"]:
            if row["overall_status"] != "error":
                validate_report(reports[row["report_dir"]], row)
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise PublicationError("The artifact could not be safely redacted.") from exc
    return summary, reports


def is_newer(candidate: dict, existing: dict | None) -> bool:
    if existing is None:
        return True
    for record in (candidate, existing):
        if not _integer(record.get("run_id"), 1) or not _integer(record.get("run_attempt"), 1):
            raise PublicationError("A retained publication receipt has an invalid run identity.")
    return (candidate["run_id"], candidate["run_attempt"]) >= (existing["run_id"], existing["run_attempt"])


def _write(path: Path, text: str) -> None:
    current = path.parent
    while True:
        if current.is_symlink():
            raise PublicationError("The site contains a symlink in an output path.")
        if current == current.parent:
            break
        current = current.parent
    if path.is_symlink():
        raise PublicationError("The site contains a symlink in an output path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dump(path: Path, data) -> None:
    _write(path, json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def _read_retained(path: Path, default):
    if path.is_symlink():
        raise PublicationError("The site contains a linked metadata file.")
    if not path.exists():
        return default
    if path.stat().st_size > MAX_FILE_BYTES:
        raise PublicationError("Retained publication metadata exceeds its size limit.")
    return _json(path.read_bytes())


def _alias(site: Path, route: str, receipt: dict) -> bool:
    existing = _read_retained(site / route / "latest.json", None)
    if not is_newer(receipt, existing):
        return False
    url = html.escape(receipt["report_url"] + "index.html", quote=True)
    _write(
        site / route / "index.html",
        f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={url}"><title>Latest SkillEvaluator report</title><p><a href="{url}">Open report</a></p>\n',
    )
    _dump(site / route / "latest.json", receipt)
    return True


def _render(directory: Path, summary: dict, context: dict) -> None:
    # The import is from the trusted helper directory, never from an artifact.
    from native_reports import render_native_reports

    errors = render_native_reports(directory, summary, context)
    expected = {row["report_dir"] for row in summary["skills"] if row["overall_status"] == "error"}
    if set(errors) - expected:
        raise PublicationError(
            "The native renderer could not faithfully publish a required report; no receipt was written."
        )


def _validate_run(run: dict, repository: str, run_id: int):
    if not isinstance(run, dict) or run.get("id") != run_id or run.get("repository", {}).get("full_name") != repository:
        raise PublicationError("The selected workflow run belongs to a different repository.")
    if run.get("path") != WORKFLOW_PATH or run.get("status") != "completed":
        raise PublicationError("Only completed Skill checks workflow runs can be published.")
    if run.get("event") not in {"push", "pull_request", "workflow_dispatch"} or run.get("conclusion") not in {
        "success",
        "failure",
        "cancelled",
        "timed_out",
        "action_required",
        "neutral",
        "skipped",
        "stale",
    }:
        raise PublicationError("The selected run has unsupported trigger or completion metadata.")
    if (
        not _integer(run.get("run_attempt"), 1)
        or not isinstance(run.get("head_sha"), str)
        or not SHA_PATTERN.fullmatch(run["head_sha"])
    ):
        raise PublicationError("The selected run lacks valid revision and attempt metadata.")


def _trusted_pins(variable: str) -> set[str]:
    raw = os.environ.get(variable, "")
    values = re.split(r"[,\n]", raw.strip())
    if not raw or len(values) > 32 or any(SHA_PATTERN.fullmatch(value.strip()) is None for value in values):
        raise PublicationError("Trusted revision allowlists require comma- or newline-separated full commit/blob SHAs.")
    return {value.strip() for value in values}


def _caller_blob(api, repository: str, sha: str) -> str:
    caller = api.get(f"/repos/{repository}/contents/{WORKFLOW_PATH}?ref={sha}")
    if (
        not isinstance(caller, dict)
        or caller.get("type") != "file"
        or caller.get("path") != WORKFLOW_PATH
        or not isinstance(caller.get("sha"), str)
        or SHA_PATTERN.fullmatch(caller["sha"]) is None
    ):
        raise PublicationError("The caller workflow lacks a valid Git blob identity.")
    return caller["sha"]


def _fork_associations(api, repository: str, run: dict) -> list:
    """Find an open fork PR from API run metadata; GitHub never records it on the run."""
    head_repository = run.get("head_repository")
    if head_repository is None:
        raise StalePublication("The fork for this PR run was deleted, so the run cannot be verified.")
    name = head_repository.get("full_name") if isinstance(head_repository, dict) else None
    if name == repository:
        raise StalePublication("The PR is closed or no longer associated with this run.")
    branch = run.get("head_branch")
    if not isinstance(name, str) or REPO_PATTERN.fullmatch(name) is None or not isinstance(branch, str) or not branch:
        raise PublicationError("The PR run lacks a valid head repository and branch.")
    head = urllib.parse.quote(name.split("/", 1)[0], safe="") + ":" + urllib.parse.quote(branch, safe="")
    pulls = api.get(f"/repos/{repository}/pulls?state=open&head={head}&per_page=100")
    if not isinstance(pulls, list) or any(not isinstance(pr, dict) for pr in pulls):
        raise PublicationError("The fork PR run has an unsupported pull request listing.")
    matches = [
        pr
        for pr in pulls
        if ((pr.get("head") or {}).get("repo") or {}).get("full_name") == name
        and ((pr.get("base") or {}).get("repo") or {}).get("full_name") == repository
    ]
    if not matches:
        raise StalePublication("The fork PR is closed or merged; no open PR uses this run's fork branch.")
    return matches


def _executed_caller_blob(api, repository: str, run: dict) -> str:
    """Reconstruct unambiguous PR merge content from GitHub-recorded revisions.

    The run API exposes the PR head, not GITHUB_WORKFLOW_SHA. Never accept a
    merge revision asserted by an artifact or the PR's current mutable ref.
    Fork runs have no recorded PR, so the PR API supplies the same revisions.
    """
    if run["event"] != "pull_request":
        return _caller_blob(api, repository, run["head_sha"])
    associations = run.get("pull_requests")
    if associations == []:
        associations = _fork_associations(api, repository, run)
    if not isinstance(associations, list) or len(associations) != 1:
        raise PublicationError("The PR run lacks an unambiguous recorded base revision; update the branch and rerun.")
    association = associations[0]
    base = (association.get("base") or {}).get("sha") if isinstance(association, dict) else None
    head = (association.get("head") or {}).get("sha") if isinstance(association, dict) else None
    if (
        not isinstance(base, str)
        or SHA_PATTERN.fullmatch(base) is None
        or not isinstance(head, str)
        or SHA_PATTERN.fullmatch(head) is None
    ):
        raise PublicationError("The PR run lacks trustworthy recorded merge revisions; update the branch and rerun.")
    if head != run["head_sha"]:
        raise StalePublication("The PR head moved after this scan.")
    head_blob = _caller_blob(api, repository, run["head_sha"])
    base_blob = _caller_blob(api, repository, base)
    if head_blob == base_blob:
        return head_blob
    comparison = api.get(f"/repos/{repository}/compare/{base}...{head}")
    ancestor = (comparison.get("merge_base_commit") or {}).get("sha") if isinstance(comparison, dict) else None
    if not isinstance(ancestor, str) or SHA_PATTERN.fullmatch(ancestor) is None:
        raise PublicationError("GitHub did not return the caller workflow merge base.")
    ancestor_blob = _caller_blob(api, repository, ancestor)
    if head_blob == ancestor_blob:
        return base_blob
    if base_blob == ancestor_blob:
        return head_blob
    raise PublicationError(
        "Both branches changed the caller workflow; update the PR branch and rerun before publishing."
    )


def _validate_workflow_provenance(api, repository: str, run: dict) -> None:
    evaluator_pins = _trusted_pins("TRUSTED_EVALUATOR_SHA")
    caller_blobs = _trusted_pins("TRUSTED_CALLER_BLOBS")
    references = run.get("referenced_workflows")
    if not isinstance(references, list) or len(references) != 1 or not isinstance(references[0], dict):
        raise PublicationError("The scan must reference exactly one approved reusable evaluator workflow.")
    reference = references[0]
    sha = reference.get("sha")
    evaluator_repository = os.environ.get("EVALUATOR_REPOSITORY", "NVIDIA/SkillEvaluator")
    if not REPO_PATTERN.fullmatch(evaluator_repository):
        raise PublicationError("The approved evaluator repository identity is invalid.")
    workflow = f"{evaluator_repository}/.github/workflows/evaluate-skills.yml"
    if sha not in evaluator_pins or reference.get("path") != f"{workflow}@{sha}":
        raise PublicationError("The selected run did not use an approved immutable evaluator workflow.")
    if _executed_caller_blob(api, repository, run) not in caller_blobs:
        raise PublicationError("The caller workflow differs from the approved workflow blobs.")


def _timestamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is None:
            raise ValueError
        return result
    except (ValueError, AttributeError) as exc:
        raise PublicationError("The run or artifact lacks a valid timestamp.") from exc


def _select_artifact(api, repository: str, run: dict) -> dict:
    listing = api.get(f"/repos/{repository}/actions/runs/{run['id']}/artifacts?per_page=100")
    if (
        not isinstance(listing, dict)
        or not isinstance(listing.get("artifacts"), list)
        or listing.get("total_count", 0) > 100
    ):
        raise PublicationError("The selected run has an unsupported artifact listing.")
    started = _timestamp(run.get("run_started_at"))
    finished = _timestamp(run.get("updated_at"))
    if finished < started:
        raise PublicationError("The selected run attempt has an invalid time interval.")
    candidates = []
    for artifact in listing["artifacts"]:
        if (
            artifact.get("name") not in {ARTIFACT_NAME, f"{ARTIFACT_NAME}-attempt-{run['run_attempt']}"}
            or artifact.get("expired") is not False
        ):
            continue
        identity = artifact.get("workflow_run", {})
        if identity.get("id") != run["id"] or identity.get("head_sha") != run["head_sha"]:
            raise PublicationError("The report artifact does not match the selected run revision.")
        if not started <= _timestamp(artifact.get("created_at")) <= finished:
            continue
        if (
            not _integer(artifact.get("id"), 1)
            or not _integer(artifact.get("size_in_bytes"))
            or artifact["size_in_bytes"] > MAX_TOTAL_BYTES
        ):
            raise PublicationError("The selected report artifact exceeds its size limit.")
        candidates.append(artifact)
    if not candidates:
        raise PublicationError("The selected run attempt has no current report artifact.")
    if len(candidates) != 1:
        raise PublicationError("The selected run attempt has ambiguous report artifacts.")
    return candidates[0]


def _current_routes(api, repository: str, run: dict) -> tuple[list[str], str]:
    sha = run["head_sha"]
    if run["event"] in {"push", "workflow_dispatch"}:
        default = api.get(f"/repos/{repository}").get("default_branch")
        if not isinstance(default, str) or run.get("head_branch") != default:
            return [], ""
        branch = api.get(f"/repos/{repository}/branches/{urllib.parse.quote(default, safe='')}")
        return (["main/latest"] if branch.get("commit", {}).get("sha") == sha else []), ""
    candidates = run.get("pull_requests") or api.get(f"/repos/{repository}/commits/{sha}/pulls")
    if not isinstance(candidates, list) or len(candidates) > 100:
        raise PublicationError("The selected run has an unsupported pull request association.")
    routes, numbers = [], []
    for candidate in candidates:
        number = candidate.get("number")
        if not _integer(number, 1):
            raise PublicationError("The selected run has an invalid pull request number.")
        pr = api.get(f"/repos/{repository}/pulls/{number}")
        if ((pr.get("base") or {}).get("repo") or {}).get("full_name") != repository:
            continue
        if ((pr.get("head") or {}).get("repo") or {}).get("full_name") != repository:
            continue
        numbers.append(str(number))
        if pr.get("state") == "open" and pr.get("head", {}).get("sha") == sha:
            routes.append(f"pr/{number}")
    return routes, numbers[0] if len(numbers) == 1 else ""


def _history_index(site: Path, history: list[dict]) -> None:
    try:
        from native_reports import _STYLE
    except ModuleNotFoundError:
        from ci.native_reports import _STYLE

    links = []
    for record in sorted(history, key=lambda item: (item["run_id"], item["run_attempt"]), reverse=True):
        url = html.escape(record["report_url"], quote=True)
        label = html.escape(
            f"Run {record['run_id']} / attempt {record['run_attempt']} — {record['overall_status']} — {record['head_sha'][:12]}"
        )
        links.append(f'<li><a href="{url}">{label}</a></li>')
    latest = []
    if (site / "main/latest/index.html").is_file():
        latest.append('<a href="main/latest/">Latest main report</a>')
    pr_root = site / "pr"
    if pr_root.is_dir() and not pr_root.is_symlink():
        for directory in sorted(pr_root.iterdir()):
            if directory.name.isdigit() and not directory.is_symlink() and (directory / "index.html").is_file():
                latest.append(f'<a href="pr/{directory.name}/">PR #{directory.name}</a>')
    _write(
        site / "index.html",
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">"
        f"<title>Report history · SkillEvaluator</title><style>{_STYLE}li{{padding:14px 0;line-height:1.6;overflow-wrap:anywhere}}ul{{padding-left:18px}}</style></head><body>"
        '<aside class="rail"><div class="brand"><span class="mark"></span>SkillEvaluator</div><p>Evaluation reports</p><a href="#history" aria-current="page">Report history</a></aside><main>'
        '<header><p class="eyebrow">Evaluation reports</p><h1>Report history</h1><p class="muted">Open the latest scan or inspect the evidence recorded for a specific commit.</p></header><section><h2>Latest reports</h2><p>'
        + " · ".join(latest)
        + '</p></section><section id="history"><h2>Retained runs</h2><ul>'
        + "".join(links)
        + "</ul></section><footer>Generated by SkillEvaluator · Each run retains its original commit and scan evidence.</footer></main></body></html>\n",
    )


def _history_identity(record: dict) -> tuple[int, int]:
    if not isinstance(record, dict) or any(not _integer(record.get(key), 1) for key in ("run_id", "run_attempt")):
        raise PublicationError("Retained site history has an invalid run identity.")
    return record["run_id"], record["run_attempt"]


def _tree_size(directory: Path) -> int:
    for ancestor in (directory, *directory.parents):
        if ancestor.is_symlink():
            raise PublicationError("Retained reports contain a linked directory.")
    if not directory.is_dir():
        raise PublicationError("A retained report directory is missing.")
    total = 0
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise PublicationError("Retained reports contain linked or special files.")
        if path.is_file():
            total += path.stat().st_size
    return total


def _prune_history(api, repository: str, site: Path, history: list[dict], current: dict, keep: int) -> list[dict]:
    """Bound deployed files while preserving current main and open-PR links.

    Git history is not rewritten. Shallow workflow checkouts download only the
    retained tree; repository administrators still own Git storage maintenance.
    """
    if not _integer(keep, 1) or keep > 1000:
        raise PublicationError("REPORT_RETENTION_RUNS must be an integer from 1 to 1000.")
    protected = {_history_identity(current)}
    main = _read_retained(site / "main/latest/latest.json", None)
    if main is not None:
        protected.add(_history_identity(main))
    pr_root = site / "pr"
    if pr_root.is_symlink():
        raise PublicationError("The PR alias directory must not be linked.")
    if pr_root.is_dir():
        aliases = list(pr_root.iterdir())
        if len(aliases) > 1000:
            raise PublicationError("The retained site has too many pull request aliases.")
        for directory in aliases:
            if not directory.name.isdecimal() or directory.is_symlink() or not directory.is_dir():
                raise PublicationError("The retained site has an unsafe pull request alias.")
            receipt = _read_retained(directory / "latest.json", None)
            if receipt is None:
                raise PublicationError("A retained pull request alias lacks its receipt.")
            pr = api.get(f"/repos/{repository}/pulls/{int(directory.name)}")
            if not isinstance(pr, dict) or pr.get("state") not in {"open", "closed"}:
                raise PublicationError("GitHub did not return a valid pull request retention state.")
            if pr["state"] == "open":
                protected.add(_history_identity(receipt))
            else:
                _tree_size(directory)
                shutil.rmtree(directory)
    records = sorted(history, key=_history_identity, reverse=True)
    sizes = {}
    for record in records:
        identity = _history_identity(record)
        if identity in sizes:
            raise PublicationError("The retained site has duplicate run identities.")
        sizes[identity] = _tree_size(site / "runs" / str(identity[0]) / str(identity[1]))
    retained = {_history_identity(record) for record in records[:keep]} | protected
    total = sum(sizes.get(identity, 0) for identity in retained)
    for record in reversed(records):
        identity = _history_identity(record)
        if total <= MAX_HISTORY_BYTES:
            break
        if identity in retained and identity not in protected:
            retained.remove(identity)
            total -= sizes[identity]
    if total > MAX_HISTORY_BYTES:
        raise PublicationError(
            "Current main/open-PR reports exceed the 512 MiB site budget; use artifacts or reduce scan scope."
        )
    for record in records:
        identity = _history_identity(record)
        if identity not in retained:
            directory = site / "runs" / str(identity[0]) / str(identity[1])
            if directory.exists():
                shutil.rmtree(directory)
    return [record for record in records if _history_identity(record) in retained]


def _recorded_image(summary: dict, fallback: str) -> str:
    execution = summary.get("execution", {})
    if not isinstance(execution, dict):
        raise PublicationError("The report's recorded execution context is invalid.")
    recorded = execution.get("image", "")
    if not isinstance(recorded, str) or not isinstance(fallback, str):
        raise PublicationError("The report's runtime image identity must be a string.")
    selected = recorded or fallback
    if selected and (
        len(selected) > 1024 or re.fullmatch(r"ghcr\.io/[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}", selected) is None
    ):
        raise PublicationError("The report's runtime image identity must be an immutable GHCR digest.")
    return selected


def publish(
    api,
    repository: str,
    run_id: int,
    base_url: str,
    site: Path,
    output: Path,
    *,
    run_attempt=1,
    render=_render,
    image="",
    retention_runs=100,
) -> dict:
    """Append an immutable report and advance only API-verified current aliases."""
    parsed = urllib.parse.urlsplit(base_url)
    if not REPO_PATTERN.fullmatch(repository) or not _integer(run_id, 1) or not _integer(run_attempt, 1):
        raise PublicationError("Repository and report run ID must be explicit valid identities.")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise PublicationError("PAGES_BASE_URL must be a plain HTTPS site URL.")
    if site.is_symlink() or output.is_symlink():
        raise PublicationError("The site and output directories must not be symlinks.")
    site.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    run = api.get(f"/repos/{repository}/actions/runs/{run_id}/attempts/{run_attempt}")
    _validate_run(run, repository, run_id)
    if run["run_attempt"] != run_attempt:
        raise PublicationError("GitHub returned a different workflow attempt from the requested one.")
    _validate_workflow_provenance(api, repository, run)
    artifact = _select_artifact(api, repository, run)
    summary, reports = read_artifact(api.download_artifact(artifact["id"]))
    successful = summary["overall_passed"] or summary["overall_status"] == "skipped"
    if (run["conclusion"] == "success") != successful:
        raise PublicationError("The workflow conclusion contradicts the report artifact verdict.")
    if summary["overall_status"] == "skipped" and summary["selection"]["head"] != run["head_sha"]:
        raise PublicationError("The no-change selection does not match the workflow revision.")
    selected_image = _recorded_image(summary, image)
    recorded_llm = summary.get("execution", {}).get("llm_enabled")
    if recorded_llm is not None and type(recorded_llm) is not bool:
        raise PublicationError("The recorded LLM setting must be a boolean when present.")
    attempt = run["run_attempt"]
    relative = f"runs/{run_id}/{attempt}"
    report_url = f"{base_url.rstrip('/')}/{relative}/"
    run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    receipt = {
        "run_id": run_id,
        "run_attempt": attempt,
        "head_sha": run["head_sha"],
        "artifact_id": artifact["id"],
        "run_url": run_url,
        "report_url": report_url,
        "overall_status": summary["overall_status"],
        "ci_conclusion": run["conclusion"],
        "event": run["event"],
        "trusted_evaluator_sha": run["referenced_workflows"][0]["sha"],
    }
    if selected_image:
        receipt["image"] = selected_image
    destination = site / relative
    existing = _read_retained(destination / "receipt.json", None)
    if existing is not None and existing != receipt:
        raise PublicationError("An immutable run path already contains different publication evidence.")
    if existing is None:
        if destination.exists():
            raise PublicationError("An immutable run path already exists without its receipt.")
        _dump(destination / "summary.json", summary)
        for directory, report in reports.items():
            _dump(destination / directory / "skillevaluator-tier1.json", report)
        render(
            destination,
            summary,
            {
                "run_url": run_url,
                "commit": run["head_sha"],
                "image": selected_image,
                "report_url": report_url,
                "llm_enabled": recorded_llm,
                "ci_conclusion": run["conclusion"],
            },
        )
        _dump(destination / "receipt.json", receipt)
    routes, pr_number = _current_routes(api, repository, run)
    updated = [_alias(site, route, receipt) for route in routes]
    current = any(updated)
    history = _read_retained(site / "history.json", [])
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise PublicationError("Retained site history is invalid.")
    history = [item for item in history if (item.get("run_id"), item.get("run_attempt")) != (run_id, attempt)]
    history.append(receipt)
    history = _prune_history(api, repository, site, history, receipt, retention_runs)
    _dump(site / "history.json", history)
    _history_index(site, history)
    _write(site / ".nojekyll", "")
    for name in ("summary.md", "pr-comment.md"):
        source = destination / name
        if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_FILE_BYTES:
            raise PublicationError("The trusted renderer did not produce required feedback files.")
        _write(output / name, source.read_text(encoding="utf-8"))
    metadata = {
        "report_url": report_url,
        "run_url": run_url,
        "pr_number": pr_number,
        "is_current": "true" if current else "false",
        "source_head_sha": run["head_sha"],
        "comment_file": str((output / "pr-comment.md").resolve()),
        "summary_file": str((output / "summary.md").resolve()),
        "run_attempt": str(attempt),
    }
    _dump(output / "publication.json", metadata)
    return metadata


def main() -> int:
    try:
        token = os.environ.get("GH_TOKEN", "").strip()
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        raw_id = os.environ.get("REPORT_RUN_ID", "")
        raw_attempt = os.environ.get("REPORT_RUN_ATTEMPT", "")
        if (
            not token
            or not raw_id.isdecimal()
            or int(raw_id) <= 0
            or not raw_attempt.isdecimal()
            or int(raw_attempt) <= 0
        ):
            raise PublicationError("GH_TOKEN and positive REPORT_RUN_ID and REPORT_RUN_ATTEMPT are required.")
        metadata = publish(
            GitHubAPI(repository, token),
            repository,
            int(raw_id),
            os.environ["PAGES_BASE_URL"],
            Path(os.environ["SITE_DIR"]),
            Path(os.environ["OUTPUT_DIR"]),
            image=os.environ.get("REPORT_IMAGE", ""),
            run_attempt=int(raw_attempt),
            retention_runs=int(os.environ.get("REPORT_RETENTION_RUNS", "100")),
        )
        if os.environ.get("GITHUB_OUTPUT"):
            with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as handle:
                for key, value in metadata.items():
                    if "\n" in value or "\r" in value:
                        raise PublicationError("A workflow output contains an unsupported newline.")
                    handle.write(f"{key}={value}\n")
        print("SkillEvaluator static reports prepared successfully.")
        return 0
    except StalePublication as exc:
        output = Path(os.environ["OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        _write(output / "summary.md", "## SkillEvaluator report skipped\n\n" + str(exc) + "\n")
        if os.environ.get("GITHUB_OUTPUT"):
            with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as handle:
                handle.write("skipped=true\n")
        print("SkillEvaluator report publication skipped: " + str(exc))
        return 0
    except PublicationError as exc:
        print("SkillEvaluator report publication rejected: " + json.dumps(str(exc)[:1000]), file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError, TypeError, urllib.error.URLError):
        print("SkillEvaluator report publication failed; no artifact code was executed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
