# SPDX-License-Identifier: Apache-2.0
"""Trusted feedback accepts API-bound current evidence and owns only its bot notes."""

from __future__ import annotations

import base64
import copy
import importlib
import io
import json
import re
import stat
import sys
import zipfile
from pathlib import Path

import pytest

REPO = "acme/skills"
SHA = "a" * 40
MERGE = "d" * 40
BLOB = "b" * 40
WORKFLOW = "NVIDIA/SkillEvaluator/.github/workflows/evaluate-all-tiers.yml@" + "c" * 40


def adapter():
    assert importlib.util.find_spec("ci.feedback") is not None, "trusted all-tier feedback adapter is required"
    return importlib.import_module("ci.feedback")


def manifest(platform="github"):
    return {
        "schema_version": 1,
        "report_contract": "skillevaluator-ci-v1",
        "commit": SHA,
        "base_commit": None,
        "repository": REPO,
        "platform": platform,
        "run_id": "123",
        "run_attempt": "1",
        "environment": "docker",
        "tiers": [
            {"tier": n, "status": "skipped", "required": n != 3, "reason": "No changed skills", "skills": []}
            for n in (1, 2, 3)
        ],
    }


def archive(payload=None, extra=None, prefix=""):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        zipped.writestr(prefix + "ci-results.json", json.dumps(payload or manifest()))
        for name, value in (extra or {}).items():
            zipped.writestr(name, value)
    return output.getvalue()


class GitHub:
    def __init__(self):
        self.run = {
            "id": 123,
            "run_attempt": 1,
            "event": "pull_request",
            "head_sha": SHA,
            "path": ".github/workflows/skill-checks-all-tiers.yml",
            "status": "completed",
            "conclusion": "success",
            "repository": {"full_name": REPO, "id": 12},
            "head_repository": {"full_name": REPO, "id": 12},
            "run_started_at": "2026-11-01T00:00:00Z",
            "updated_at": "2026-11-01T00:10:00Z",
            "pull_requests": [{"number": 7}],
            "referenced_workflows": [{"path": WORKFLOW, "sha": "c" * 40}],
        }
        self.pr = {
            "state": "open",
            "number": 7,
            "head": {"sha": SHA, "repo": {"full_name": REPO}},
            "base": {"repo": {"full_name": REPO}},
        }
        self.artifacts = [
            {
                "id": n,
                "name": name,
                "expired": False,
                "size_in_bytes": 100,
                "created_at": "2026-11-01T00:05:00Z",
                "workflow_run": {"id": 123, "head_sha": SHA},
            }
            for n, name in (
                (21, "skill-evaluator-results-attempt-1"),
                (22, "skill-evaluator-results-provenance-attempt-1"),
            )
        ]
        self.payload = archive()
        self.comments = []
        self.writes = []
        self.gets = []
        self.downloads = []
        self.other_runs = {}
        self.flip_head_on_recheck = False

    def get(self, path):
        self.gets.append(path)
        prefix = "/repos/" + REPO
        if path == prefix + "/actions/runs/123":
            return copy.deepcopy(self.run)
        if path.startswith(prefix + "/actions/runs/") and path.rsplit("/", 1)[-1].isdigit():
            identifier = int(path.rsplit("/", 1)[-1])
            if identifier not in self.other_runs:
                import urllib.error

                raise urllib.error.HTTPError("https://api.github.com" + path, 404, "Not Found", {}, None)
            return copy.deepcopy(self.other_runs[identifier])
        if path == prefix + "/actions/runs/123/artifacts?per_page=100":
            return {"total_count": len(self.artifacts), "artifacts": copy.deepcopy(self.artifacts)}
        if path.startswith(prefix + "/contents/"):
            assert path.endswith("?ref=" + MERGE)
            return {"type": "file", "path": ".github/workflows/skill-checks-all-tiers.yml", "sha": BLOB}
        if path == prefix + "/pulls/7":
            pr = copy.deepcopy(self.pr)
            if self.flip_head_on_recheck and self.gets.count(path) > 1:
                pr["head"]["sha"] = "e" * 40
            return pr
        if path.startswith(prefix + "/issues/7/comments?"):
            return copy.deepcopy(self.comments)
        if path.startswith(prefix + "/issues/comments/"):
            identifier = int(path.rsplit("/", 1)[-1])
            return copy.deepcopy(next(c for c in self.comments if c["id"] == identifier))
        raise AssertionError(path)

    def download_artifact(self, identifier):
        self.downloads.append(identifier)
        return self.payload if identifier == 21 else b"signed-receipt"

    def post(self, path, payload):
        self.writes.append(("POST", path, payload))
        comment = {
            "id": 51,
            "body": payload["body"],
            "user": {"login": "github-actions[bot]", "type": "Bot"},
            "html_url": "https://github.com/acme/skills/pull/7#issuecomment-51",
        }
        self.comments.append(comment)
        return copy.deepcopy(comment)

    def patch(self, path, payload):
        self.writes.append(("PATCH", path, payload))
        comment = next(c for c in self.comments if c["id"] == int(path.rsplit("/", 1)[-1]))
        comment.update(payload)
        return copy.deepcopy(comment)


@pytest.fixture
def github(monkeypatch):
    module = adapter()
    monkeypatch.setenv("SKILL_EVAL_TRUSTED_WORKFLOW_REF", WORKFLOW)
    monkeypatch.setenv("SKILL_EVAL_TRUSTED_CALLER_BLOBS", BLOB)

    def verify(raw, **kw):
        assert raw == b"signed-receipt"
        assert kw["evaluator_workflow"] + "@" + kw["evaluator_sha"] == WORKFLOW
        assert kw["caller_workflow_path"] == ".github/workflows/skill-checks-all-tiers.yml"
        return {"workflow_sha": MERGE}

    monkeypatch.setattr(module, "verify_receipt", verify)
    return module, GitHub()


def test_github_creates_one_owned_comment_with_private_artifact_and_gate(github):
    module, api = github
    result = module.github_feedback(api, REPO, 123, 1)
    assert result["comment_action"] == "created"
    body = api.comments[0]["body"]
    assert module.MARKER in body
    assert "https://github.com/acme/skills/actions/runs/123/artifacts/21" in body
    assert "Required tiers: Tier 1, Tier 2. Advisory tiers: Tier 3." in body
    assert body.count("SKIPPED") == 3
    assert body.count("[Report](") == 1
    assert "(tier" not in body  # archive members are not live web pages
    assert module.github_feedback(api, REPO, 123, 1)["comment_action"] == "unchanged"
    assert len(api.writes) == 1


@pytest.mark.parametrize("platform", ["github", "gitlab"])
@pytest.mark.parametrize("tier1_status", ["passed", "incomplete"])
def test_trusted_feedback_keeps_paid_failures_advisory_and_tier1_blocking(request, platform, tier1_status):
    from skillevaluator.models import ValidationResult
    from skillevaluator.reporting.json_reporter import JSONReporter

    module, api = request.getfixturevalue(platform)
    payload = manifest(platform)
    payload["tiers"][0].update(status=tier1_status, reason="Tier 1 validation")
    extra = {}
    if tier1_status == "passed":
        native = JSONReporter().render_all(
            [ValidationResult(validator_name="FORMAT", metadata={"gating": {"tier": 1, "blocking": True}})]
        )
        payload["tiers"][0]["skills"] = [
            {
                "path": "skills/demo",
                "status": "passed",
                "report_json": "tier1/skill-001/skillevaluator-tier1.json",
                "report_html": "tier1/skill-001/skillevaluator-tier1.html",
            }
        ]
        extra = {
            "tier1/skill-001/skillevaluator-tier1.json": native,
            "tier1/skill-001/skillevaluator-tier1.html": "<p>Tier 1 passed</p>",
        }
    payload["tiers"][1].update(status="failed", required=False, reason="Duplication check failed")
    payload["tiers"][2].update(status="incomplete", required=False, reason="Paired evaluation timed out")
    prefix = "skill-evaluator-results/" if platform == "gitlab" else ""
    api.payload = archive(payload, {prefix + key: value for key, value in extra.items()}, prefix=prefix)

    retained = module.read_results(api.payload, prefix=prefix)
    assert retained["overall_passed"] is (tier1_status == "passed")
    if platform == "github":
        result = module.github_feedback(api, REPO, 123, 1)
        body = api.comments[0]["body"]
        report_url = "https://github.com/acme/skills/actions/runs/123/artifacts/21"
    else:
        result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
        body = api.notes[0]["body"]
        report_url = "https://gitlab.example/acme/skills/-/jobs/91/artifacts/download"
    assert result["comment_action"] == "created"
    assert f"Required gate: **{'Passed' if tier1_status == 'passed' else 'Not passed'}**" in body
    assert tier1_status.upper() in body
    assert "FAILED" in body and "Duplication check failed" in body
    assert "INCOMPLETE" in body and "Paired evaluation timed out" in body
    assert "Only Tier 1 blocks CI; Tier 2 and Tier 3 are advisory." in body
    assert report_url in body


def compact_archive(platform):
    from skillevaluator.models import Finding, Severity, ValidationResult
    from skillevaluator.reporting.json_reporter import JSONReporter

    native = ValidationResult(validator_name="Security", metadata={"gating": {"tier": 1, "blocking": True}})
    for severity, message in (
        (Severity.LOW, "Missing optional metadata"),
        (Severity.CRITICAL, "Exposed credential pattern"),
        (Severity.HIGH, "Unsafe shell command"),
        (Severity.INFO, "Static checks completed"),
    ):
        native.add_finding(
            Finding(category="SECURITY", severity=severity, check_name="fixture", message=message, file_path="SKILL.md")
        )
    payload = manifest(platform)
    for tier in payload["tiers"]:
        tier.update(
            status="failed" if tier["tier"] == 1 else "incomplete",
            required=tier["tier"] == 1,
            reason="Validation findings" if tier["tier"] == 1 else "Missing provider credentials",
            skills=[
                {
                    "path": "skills/demo",
                    "status": "failed" if tier["tier"] == 1 else "incomplete",
                    "report_json": None,
                    "report_html": None,
                    # Producer-supplied counters are not trusted comment evidence.
                    "metrics": {"issue_count": 0, "issue_count_known": True, "critical_count": 0},
                }
            ],
        )
    skill = payload["tiers"][0]["skills"][0]
    skill.update(
        report_json="tier1/skill-001/skillevaluator-tier1.json",
        report_html="tier1/skill-001/skillevaluator-tier1.html",
    )
    prefix = "skill-evaluator-results/" if platform == "gitlab" else ""
    extra = {
        prefix + skill["report_json"]: JSONReporter().render_all([native]),
        prefix + skill["report_html"]: "<p>Complete retained Tier 1 findings</p>",
    }
    return archive(payload, extra, prefix=prefix)


@pytest.mark.parametrize("platform", ["github", "gitlab"])
def test_trusted_comment_has_one_combined_report_link_and_compact_verified_skill_summary(request, platform):
    module, api = request.getfixturevalue(platform)
    api.payload = compact_archive(platform)
    if platform == "github":
        result = module.github_feedback(api, REPO, 123, 1)
        body = api.comments[0]["body"]
        report_url = "https://github.com/acme/skills/actions/runs/123/artifacts/21"
    else:
        result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
        body = api.notes[0]["body"]
        report_url = "https://gitlab.example/acme/skills/-/jobs/91/artifacts/download"
    assert result["comment_action"] == "created"
    assert body.count("| Skill |") == 1
    assert body.count("[Report](" + report_url + ")") == 1
    assert body.count(report_url) == 1
    assert all(name in body for name in ("Static and Security Validation", "Deduplication", "Live Agent Evaluation"))
    skill_row = next(line for line in body.splitlines() if line.startswith("| skills/demo |"))
    assert "FAILED" in skill_row and "3 issues" in skill_row
    assert "1 critical" in skill_row and "1 high" in skill_row
    assert "Exposed credential pattern" in skill_row
    assert skill_row.count("INCOMPLETE") == 2
    assert skill_row.lower().count("issues unknown") == 2
    assert "skill evidence" not in body and "In downloadable artifact" not in body
    assert "Required gate: **Not passed**" in body


@pytest.mark.parametrize("platform", ["github", "gitlab"])
def test_feedback_issue_counts_are_derived_from_native_evidence_and_unknown_without_it(platform):
    module = adapter()
    prefix = "skill-evaluator-results/" if platform == "gitlab" else ""
    retained = module.read_results(compact_archive(platform), prefix=prefix)
    metrics = retained["tiers"][0]["skills"][0]["metrics"]
    assert metrics["issue_count"] == 3
    assert metrics["critical_count"] == 1
    assert metrics["high_count"] == 1
    assert metrics["issue_count_known"] is True
    for tier in retained["tiers"][1:]:
        assert tier["skills"][0].get("metrics", {}).get("issue_count_known", False) is False


@pytest.mark.parametrize(
    "change", ["attempt", "revision", "expired", "duplicate", "caller", "workflow", "head", "foreign"]
)
def test_github_rejects_untrusted_or_stale_evidence_before_write(github, monkeypatch, change):
    module, api = github
    if change == "attempt":
        api.run["run_attempt"] = 2
    if change == "revision":
        payload = manifest()
        payload["commit"] = "e" * 40
        api.payload = archive(payload)
    if change == "expired":
        api.artifacts[0]["expired"] = True
    if change == "duplicate":
        api.artifacts.append(copy.deepcopy(api.artifacts[0]))
    if change == "caller":
        monkeypatch.setenv("SKILL_EVAL_TRUSTED_CALLER_BLOBS", "f" * 40)
    if change == "workflow":
        api.run["referenced_workflows"][0]["sha"] = "e" * 40
    if change == "head":
        api.pr["head"]["sha"] = "e" * 40
    if change == "foreign":
        api.pr["head"]["repo"]["full_name"] = "fork/skills"
    try:
        result = module.github_feedback(api, REPO, 123, 1)
        assert result["comment_action"] == "skipped"
    except module.FeedbackError:
        pass
    assert not api.writes


def test_github_rechecks_head_after_comments_and_preserves_human_marker(github):
    module, api = github
    api.comments = [{"id": 5, "body": module.MARKER, "user": {"login": "someone", "type": "User"}}]
    api.flip_head_on_recheck = True
    assert module.github_feedback(api, REPO, 123, 1)["comment_action"] == "skipped"
    assert not api.writes


def test_github_does_not_overwrite_newer_run_receipt(github):
    module, api = github
    api.other_runs[124] = {**copy.deepcopy(api.run), "id": 124}
    api.comments = [
        {
            "id": 5,
            "body": module.MARKER + "\n<!-- skillevaluator-all-tiers-run:124:1 -->",
            "user": {"login": "github-actions[bot]", "type": "Bot"},
        }
    ]
    assert module.github_feedback(api, REPO, 123, 1)["comment_action"] == "skipped"
    assert not api.writes


def _bot_comment(module, identifier, marker):
    body = module.MARKER + f"\n<!-- skillevaluator-all-tiers-run:{marker} -->\nRequired gate: **Passed**"
    return {"id": identifier, "body": body, "user": {"login": "github-actions[bot]", "type": "Bot"}}


@pytest.mark.parametrize(
    "marker",
    [
        "99999999999:1",  # no such run
        "124:1",  # another workflow's run
        "125:1",  # producer run for a different PR head
        "126:1",  # producer run from another event
        "123:9",  # attempt that never existed
    ],
)
def test_github_overwrites_unverifiable_bot_marker_instead_of_deferring(github, marker):
    module, api = github
    api.other_runs[124] = {**copy.deepcopy(api.run), "id": 124, "path": ".github/workflows/other.yml"}
    api.other_runs[125] = {**copy.deepcopy(api.run), "id": 125, "head_sha": "e" * 40}
    api.other_runs[126] = {**copy.deepcopy(api.run), "id": 126, "event": "push"}
    api.comments = [_bot_comment(module, 5, marker)]
    result = module.github_feedback(api, REPO, 123, 1)
    assert result["comment_action"] == "updated"
    assert [(method, path) for method, path, _ in api.writes] == [("PATCH", f"/repos/{REPO}/issues/comments/5")]
    assert "<!-- skillevaluator-all-tiers-run:123:1 -->" in api.comments[0]["body"]
    assert "Required gate: **Passed**" not in api.comments[0]["body"]


def test_github_multiple_unverifiable_bot_markers_do_not_block_genuine_feedback(github):
    module, api = github
    api.comments = [
        _bot_comment(module, 5, "99999999999:1"),
        _bot_comment(module, 6, "99999999998:1"),
        {**_bot_comment(module, 7, "1:1"), "body": module.MARKER + "\nno producer receipt"},
    ]
    assert module.github_feedback(api, REPO, 123, 1)["comment_action"] == "updated"
    assert [path for _, path, _ in api.writes] == [f"/repos/{REPO}/issues/comments/5"]
    # A verified genuine comment is preferred over leftover unverifiable ones.
    assert module.github_feedback(api, REPO, 123, 1)["comment_action"] == "unchanged"
    assert len(api.writes) == 1


def test_github_marker_verification_does_not_hide_api_failures(github):
    import urllib.error

    module, api = github
    api.comments = [_bot_comment(module, 5, "124:1")]
    original = api.get

    def unavailable(path):
        if path.endswith("/actions/runs/124"):
            raise urllib.error.HTTPError("https://api.github.com" + path, 503, "Unavailable", {}, None)
        return original(path)

    api.get = unavailable
    with pytest.raises(urllib.error.HTTPError):
        module.github_feedback(api, REPO, 123, 1)
    assert not api.writes


@pytest.mark.parametrize(
    "change",
    ["fork", "deleted-fork", "rerun-started", "rerun-completed", "rerun-in-progress"],
)
def test_github_skips_fork_and_superseded_producer_runs_without_failing(github, change):
    module, api = github
    if change == "fork":
        api.run["head_repository"] = {"full_name": "contributor/skills", "id": 99}
        # Fork runs have no OIDC token, so no provenance receipt is uploaded.
        api.artifacts = [a for a in api.artifacts if "provenance" not in a["name"]]
    elif change == "deleted-fork":
        api.run["head_repository"] = None
    elif change == "rerun-started":
        api.run.update(run_attempt=2, status="in_progress")
    elif change == "rerun-completed":
        api.run["run_attempt"] = 2
    else:
        api.run["status"] = "in_progress"
    result = module.github_feedback(api, REPO, 123, 1)
    assert result["comment_action"] == "skipped"
    assert not api.writes and not api.downloads
    assert not any("/artifacts" in path for path in api.gets)


def test_github_feedback_job_skips_fork_pull_request_runs():
    import yaml

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/feedback-all-tiers.yml").read_text())
    condition = workflow["jobs"]["feedback"]["if"]
    assert "github.event.workflow_run.event == 'pull_request'" in condition
    assert "github.event.workflow_run.head_repository.full_name == github.repository" in condition


@pytest.mark.parametrize("name", ["../ci-results.json", "/ci-results.json", "tier1/../../secret.json", "evil\\file"])
def test_zip_paths_are_rejected_without_extraction(name):
    module = adapter()
    with pytest.raises(module.FeedbackError):
        module.read_results(archive(extra={name: "{}"}))


def test_zip_symlinks_duplicate_json_and_missing_native_evidence_are_rejected():
    module = adapter()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        link = zipfile.ZipInfo("ci-results.json")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        zipped.writestr(link, json.dumps(manifest()))
    with pytest.raises(module.FeedbackError):
        module.read_results(output.getvalue())
    payload = manifest()
    payload["tiers"][0].update(
        status="passed",
        skills=[
            {
                "path": "skills/demo",
                "status": "passed",
                "report_json": "tier1/skill-001/skillevaluator-tier1.json",
                "report_html": "tier1/skill-001/skillevaluator-tier1.html",
            }
        ],
    )
    with pytest.raises(module.FeedbackError):
        module.read_results(archive(payload))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        zipped.writestr("ci-results.json", '{"schema_version":1,"schema_version":1}')
    with pytest.raises(module.FeedbackError):
        module.read_results(output.getvalue())


def config_content():
    import yaml

    root = {
        "include": [
            {
                "remote": "https://raw.githubusercontent.com/NVIDIA/SkillEvaluator/"
                + "c" * 40
                + "/ci/gitlab/all-tiers.yml"
            },
            {
                "local": ".gitlab/skill-feedback.yml",
                "rules": [
                    {
                        "if": '$CI_PIPELINE_SOURCE == "pipeline" && $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH && $SKILL_EVAL_FEEDBACK == "true"'
                    }
                ],
            },
        ]
    }
    return {"encoding": "base64", "content": base64.b64encode(yaml.safe_dump(root).encode()).decode()}


class GitLab:
    def __init__(self):
        self.project = {
            "id": 12,
            "path_with_namespace": REPO,
            "default_branch": "main",
            "ci_config_path": None,
            "web_url": "https://gitlab.example/acme/skills",
        }
        self.pipeline = {"id": 123, "sha": SHA, "ref": "feature", "source": "merge_request_event", "status": "success"}
        self.publisher = {"id": 456, "sha": MERGE, "ref": "main", "source": "pipeline", "status": "running"}
        self.job = {
            "id": 91,
            "name": "skill-evaluation-aggregate",
            "status": "success",
            "retried": False,
            "pipeline": {"id": 123},
            "commit": {"id": SHA},
            "ref": "feature",
            "artifacts_file": {"filename": "skill-evaluator-results-123-91.zip", "size": 1000},
        }
        self.mr = {
            "iid": 7,
            "state": "opened",
            "source_project_id": 12,
            "target_project_id": 12,
            "sha": SHA,
            "source_branch": "feature",
            "head_pipeline": {"id": 123},
        }
        self.request_job = {
            "id": 93,
            "name": "skill-evaluation-request-feedback",
            "status": "success",
            "retried": False,
            "pipeline": {"id": 123},
            "commit": {"id": SHA},
            "ref": "feature",
        }
        self.relationship = {
            "project": {
                "fullPath": REPO,
                "pipeline": {
                    "id": "gid://gitlab/Ci::Pipeline/456",
                    "project": {"fullPath": REPO},
                    "sourceJob": {"id": "gid://gitlab/Ci::Build/93"},
                    "upstream": {
                        "id": "gid://gitlab/Ci::Pipeline/123",
                        "project": {"fullPath": REPO},
                        "mergeRequest": {"iid": "7", "sourceProjectId": 12, "targetProjectId": 12},
                    },
                },
            }
        }
        self.current_request_id = 93
        self.notes = []
        self.writes = []
        self.payload = archive(manifest("gitlab"), prefix="skill-evaluator-results/")
        self.user = {"id": 77, "username": "project_12_bot_abc", "bot": True}
        self.source_blob = BLOB
        self.flip_head_on_recheck = False
        self.head_checks = 0
        self.variables = [
            {"key": "SKILL_EVAL_FEEDBACK", "value": "true", "variable_type": "env_var"},
            {"key": "SKILL_EVAL_SOURCE_PIPELINE_ID", "value": "123", "variable_type": "env_var"},
        ]

    def get(self, path):
        if path == "/user":
            return copy.deepcopy(self.user)
        if path in {"/projects/acme%2Fskills", "/projects/12"}:
            return copy.deepcopy(self.project)
        root = "/projects/12"
        if path == root + "/pipelines/123":
            return copy.deepcopy(self.pipeline)
        if path == root + "/pipelines/456":
            return copy.deepcopy(self.publisher)
        if path == root + "/pipelines/456/variables":
            return copy.deepcopy(self.variables)
        if path == root + "/pipelines/123/variables":
            return []
        if path == root + "/repository/branches/main":
            return {"name": "main", "protected": True, "commit": {"id": MERGE}}
        if path == root + "/jobs/92":
            return {
                "id": 92,
                "name": "skill-evaluation-feedback",
                "status": "running",
                "ref": "main",
                "pipeline": {"id": 456},
                "commit": {"id": MERGE},
            }
        if path == root + "/jobs/93":
            return copy.deepcopy(self.request_job)
        if path.startswith(root + "/pipelines/123/jobs?"):
            request = {**self.request_job, "id": self.current_request_id}
            return [copy.deepcopy(self.job), copy.deepcopy(request)]
        if path == root + "/repository/files/.gitlab-ci.yml?ref=" + SHA:
            return {"file_path": ".gitlab-ci.yml", "blob_id": self.source_blob, "commit_id": SHA, **config_content()}
        if path == root + "/repository/files/.gitlab-ci.yml?ref=" + MERGE:
            return {"file_path": ".gitlab-ci.yml", "blob_id": BLOB, "commit_id": MERGE, **config_content()}
        if path == root + "/merge_requests/7":
            mr = copy.deepcopy(self.mr)
            self.head_checks += 1
            if self.flip_head_on_recheck and self.head_checks > 1:
                mr["sha"] = "e" * 40
            return mr
        if path.startswith(root + "/merge_requests/7/notes?"):
            return copy.deepcopy(self.notes)
        if path.startswith(root + "/merge_requests/7/notes/"):
            return copy.deepcopy(next(n for n in self.notes if n["id"] == int(path.rsplit("/", 1)[-1])))
        raise AssertionError(path)

    def graphql(self, query, variables):
        assert "sourceJob" in query and "upstream" in query
        assert "mergeRequest{iid sourceProjectId targetProjectId}" in query
        assert variables == {"project": REPO, "pipeline": "gid://gitlab/Ci::Pipeline/456"}
        return copy.deepcopy(self.relationship)

    def download_artifact(self, job):
        assert job == 91
        return self.payload

    def post(self, path, payload):
        self.writes.append(("POST", path, payload))
        note = {"id": 51, "body": payload["body"], "author": copy.deepcopy(self.user)}
        self.notes.append(note)
        return copy.deepcopy(note)

    def put(self, path, payload):
        self.writes.append(("PUT", path, payload))
        note = next(n for n in self.notes if n["id"] == int(path.rsplit("/", 1)[-1]))
        note.update(payload)
        return copy.deepcopy(note)


@pytest.fixture
def gitlab(monkeypatch):
    module = adapter()

    def verify(api_url, token):
        assert api_url == "https://gitlab.example/api/v4" and token == "signed-job-token"
        return {"project_id": "12", "pipeline_id": "456", "job_id": "92", "ref": "main", "sha": MERGE}

    monkeypatch.setenv("SKILL_EVAL_GITLAB_ID_TOKEN", "signed-job-token")
    monkeypatch.setattr(module, "verify_gitlab_job", verify, raising=False)
    return module, GitLab()


def test_gitlab_binds_protected_downstream_source_job_config_head_job_and_bot(gitlab):
    module, api = gitlab
    result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "acme/skills", 123)
    assert result["comment_action"] == "created"
    body = api.notes[0]["body"]
    assert "https://gitlab.example/acme/skills/-/jobs/91/artifacts/download" in body
    assert "https://gitlab.example/acme/skills/-/pipelines/123" in body
    assert module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)["comment_action"] == "unchanged"
    assert len(api.writes) == 1


@pytest.mark.parametrize("action", ["created", "updated"])
def test_gitlab_sends_note_without_the_final_lf_removed_by_server(gitlab, monkeypatch, action):
    module, api = gitlab
    if action == "updated":
        module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
        api.notes[0]["body"] += "\nold content"
        api.writes.clear()

    def normalize_write(write):
        def store(path, payload):
            response = write(path, payload)
            note = next(note for note in api.notes if note["id"] == response["id"])
            note["body"] = note["body"].removesuffix("\n")
            return {**response, "body": note["body"]}

        return store

    monkeypatch.setattr(api, "post", normalize_write(api.post))
    monkeypatch.setattr(api, "put", normalize_write(api.put))
    result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert result["comment_action"] == action
    assert api.writes[0][2]["body"] == api.notes[0]["body"]
    assert not api.notes[0]["body"].endswith("\n")
    assert module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)["comment_action"] == "unchanged"
    assert len(api.writes) == 1


@pytest.mark.parametrize(
    "payload",
    [
        '{"id":"12","ref":"main","variables":{"SKILL_EVAL_FEEDBACK":"true","SKILL_EVAL_SOURCE_PIPELINE_ID":"123"}}',
        "opaque content: not JSON; $(do-not-execute)",
    ],
)
def test_gitlab_accepts_server_added_file_trigger_payload_without_using_its_contents(gitlab, payload):
    module, api = gitlab
    api.variables.append({"key": "TRIGGER_PAYLOAD", "value": payload, "variable_type": "file"})
    result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert result["comment_action"] == "created"
    assert payload not in api.notes[0]["body"]


@pytest.mark.parametrize(
    "change",
    [
        "payload-env",
        "payload-type-missing",
        "payload-nonstring",
        "payload-oversized",
        "payload-duplicate",
        "selector-duplicate",
        "selector-file",
        "selector-type-missing",
        "extra-env",
        "feedback-value",
        "source-value",
        "invalid-key",
    ],
)
def test_gitlab_rejects_extra_duplicate_or_mistyped_trigger_variables(gitlab, change):
    module, api = gitlab
    payload = {"key": "TRIGGER_PAYLOAD", "value": "opaque server payload", "variable_type": "file"}
    api.variables.append(payload)
    if change == "payload-env":
        payload["variable_type"] = "env_var"
    elif change == "payload-type-missing":
        payload.pop("variable_type")
    elif change == "payload-nonstring":
        payload["value"] = {"ref": "main"}
    elif change == "payload-oversized":
        payload["value"] = "x" * (64 * 1024 + 1)
    elif change == "payload-duplicate":
        api.variables.append(copy.deepcopy(payload))
    elif change == "selector-duplicate":
        api.variables[-1] = copy.deepcopy(api.variables[0])
    elif change == "selector-file":
        api.variables[0]["variable_type"] = "file"
        api.variables.pop()
    elif change == "selector-type-missing":
        api.variables[0].pop("variable_type")
        api.variables.pop()
    elif change == "extra-env":
        api.variables[-1] = {"key": "CI_PIPELINE_ID", "value": "123", "variable_type": "env_var"}
    elif change == "feedback-value":
        api.variables[0]["value"] = "false"
    elif change == "source-value":
        api.variables[1]["value"] = "124"
    else:
        payload["key"] = []
    with pytest.raises(module.FeedbackError, match="forwarded variables"):
        module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert not api.writes


@pytest.mark.parametrize("ref", ["ci/skillevaluator-all-tiers-setup", "refs/merge-requests/7/head"])
def test_gitlab_accepts_current_mr_branch_or_exact_detached_head_ref(gitlab, ref):
    module, api = gitlab
    api.mr["source_branch"] = "ci/skillevaluator-all-tiers-setup"
    api.pipeline["ref"] = api.job["ref"] = api.request_job["ref"] = ref
    result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert result["comment_action"] == "created"
    assert len(api.writes) == 1


@pytest.mark.parametrize(
    "ref",
    [
        "refs/merge-requests/8/head",
        "refs/merge-requests/7/merge",
        "refs/merge-requests/07/head",
        "refs/merge-requests/7/head/other",
        "other-branch",
    ],
)
def test_gitlab_rejects_unrelated_or_non_head_mr_producer_refs(gitlab, ref):
    module, api = gitlab
    api.pipeline["ref"] = api.job["ref"] = api.request_job["ref"] = ref
    result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert result["comment_action"] == "skipped"
    assert not api.writes


@pytest.mark.parametrize("branch", [None, "", ["feature"]])
def test_gitlab_detached_head_ref_still_requires_a_source_branch(gitlab, branch):
    module, api = gitlab
    api.pipeline["ref"] = api.job["ref"] = api.request_job["ref"] = "refs/merge-requests/7/head"
    api.mr["source_branch"] = branch
    result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert result["comment_action"] == "skipped"
    assert not api.writes


@pytest.mark.parametrize("change", ["head", "head-pipeline", "job-ref", "recheck"])
def test_gitlab_detached_head_ref_keeps_exact_current_producer_guards(gitlab, change):
    module, api = gitlab
    api.pipeline["ref"] = api.job["ref"] = api.request_job["ref"] = "refs/merge-requests/7/head"
    if change == "head":
        api.mr["sha"] = "e" * 40
    elif change == "head-pipeline":
        api.mr["head_pipeline"]["id"] = 124
    elif change == "job-ref":
        api.job["ref"] = "refs/merge-requests/8/head"
    else:
        api.flip_head_on_recheck = True
    if change == "job-ref":
        with pytest.raises(module.FeedbackError, match="aggregate job artifact identity"):
            module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    else:
        result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
        assert result["comment_action"] == "skipped"
    assert not api.writes


def test_gitlab_accepts_running_source_when_request_job_and_aggregate_are_bound(gitlab):
    module, api = gitlab
    api.pipeline["status"] = api.request_job["status"] = "running"
    result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert result["comment_action"] == "created"


@pytest.mark.parametrize("status", ["manual", "pending", "created", "waiting_for_resource", "preparing", "scheduled"])
def test_gitlab_accepts_unfinished_source_pipeline_when_other_consumer_jobs_remain(gitlab, status):
    module, api = gitlab
    # Merged consumer CI can leave later or blocking manual jobs after the
    # finished aggregate and request jobs, which carry the bound evidence.
    api.pipeline["status"] = status
    result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert result["comment_action"] == "created"


def test_gitlab_uses_persisted_graphql_mr_without_unsupported_pipeline_rest_endpoint(gitlab):
    import urllib.error

    module, api = gitlab
    original = api.get

    def historical_api(path):
        if path == "/projects/12/pipelines/123/merge_requests":
            raise urllib.error.HTTPError("https://gitlab.example/api/v4" + path, 404, "Not found", {}, None)
        return original(path)

    api.get = historical_api
    result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert result["comment_action"] == "created"


@pytest.mark.parametrize(
    "linked_mr",
    [
        None,
        [],
        {},
        {"iid": 7, "sourceProjectId": 12, "targetProjectId": 12},
        {"iid": "0", "sourceProjectId": 12, "targetProjectId": 12},
        {"iid": "07", "sourceProjectId": 12, "targetProjectId": 12},
        {"iid": "-7", "sourceProjectId": 12, "targetProjectId": 12},
        {"iid": "7/head", "sourceProjectId": 12, "targetProjectId": 12},
        {"iid": "9223372036854775808", "sourceProjectId": 12, "targetProjectId": 12},
        {"iid": "7", "sourceProjectId": 13, "targetProjectId": 12},
        {"iid": "7", "sourceProjectId": 12, "targetProjectId": 13},
        {"iid": "7", "sourceProjectId": "12", "targetProjectId": 12},
        {"iid": "7", "sourceProjectId": 12, "targetProjectId": "12"},
    ],
)
def test_gitlab_rejects_missing_malformed_or_cross_project_persisted_mr(gitlab, linked_mr):
    module, api = gitlab
    api.relationship["project"]["pipeline"]["upstream"]["mergeRequest"] = linked_mr
    with pytest.raises(module.FeedbackError, match="linked same-project merge request"):
        module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert not api.writes


@pytest.mark.parametrize(
    "change",
    [
        "project",
        "publisher",
        "publisher-project",
        "upstream",
        "upstream-project",
        "missing-job",
        "bridge-job",
        "job-name",
        "job-pipeline",
        "job-sha",
        "job-ref",
        "retried",
        "not-current",
        "aggregate-running",
    ],
)
def test_gitlab_api_trigger_requires_server_bound_current_request_job(gitlab, change):
    module, api = gitlab
    relation = api.relationship["project"]["pipeline"]
    if change == "project":
        api.relationship["project"]["fullPath"] = "other/project"
    elif change == "publisher":
        relation["id"] = "gid://gitlab/Ci::Pipeline/457"
    elif change == "publisher-project":
        relation["project"]["fullPath"] = "other/project"
    elif change == "upstream":
        relation["upstream"]["id"] = "gid://gitlab/Ci::Pipeline/124"
    elif change == "upstream-project":
        relation["upstream"]["project"]["fullPath"] = "other/project"
    elif change == "missing-job":
        relation["sourceJob"] = None
    elif change == "bridge-job":
        relation["sourceJob"]["id"] = "gid://gitlab/Ci::Bridge/93"
    elif change == "job-name":
        api.request_job["name"] = "untrusted-job"
    elif change == "job-pipeline":
        api.request_job["pipeline"]["id"] = 124
    elif change == "job-sha":
        api.request_job["commit"]["id"] = "e" * 40
    elif change == "job-ref":
        api.request_job["ref"] = "other"
    elif change == "retried":
        api.request_job["retried"] = True
    elif change == "not-current":
        api.current_request_id = 94
    else:
        api.job["status"] = "running"
    with pytest.raises(module.FeedbackError):
        module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert not api.writes


def test_gitlab_rechecks_request_job_after_artifact_download(gitlab):
    module, api = gitlab
    original = api.download_artifact

    def replaced(job):
        payload = original(job)
        api.current_request_id = 94
        return payload

    api.download_artifact = replaced
    with pytest.raises(module.FeedbackError):
        module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert not api.writes


@pytest.mark.parametrize(
    "change",
    [
        "relationship",
        "producer-config",
        "ref",
        "fork",
        "head",
        "job-ref",
        "job-name",
        "retried",
        "head-pipeline",
        "bot",
        "variable-override",
    ],
)
def test_gitlab_rejects_forged_producer_or_privileged_context(gitlab, change):
    module, api = gitlab
    if change == "relationship":
        api.relationship["project"]["pipeline"]["upstream"]["id"] = "gid://gitlab/Ci::Pipeline/124"
    if change == "producer-config":
        api.source_blob = "e" * 40
    if change == "ref":
        api.publisher["ref"] = "feature"
    if change == "fork":
        api.mr["source_project_id"] = 13
    if change == "head":
        api.mr["sha"] = "e" * 40
    if change == "job-ref":
        api.job["ref"] = "other"
    if change == "job-name":
        api.job["name"] = "evil-artifact"
    if change == "retried":
        api.job["retried"] = True
    if change == "head-pipeline":
        api.mr["head_pipeline"]["id"] = 124
    if change == "bot":
        api.user["bot"] = False
    if change == "variable-override":
        api.variables.append({"key": "CI_PIPELINE_ID", "value": "123"})
    try:
        result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
        assert result["comment_action"] == "skipped"
    except module.FeedbackError:
        pass
    assert not api.writes


def test_gitlab_rechecks_head_and_ignores_human_or_other_bot_marker(gitlab):
    module, api = gitlab
    api.notes = [{"id": 5, "body": module.MARKER, "author": {"id": 78, "bot": True}}]
    api.flip_head_on_recheck = True
    assert module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)["comment_action"] == "skipped"
    assert not api.writes


@pytest.mark.parametrize(
    "url",
    [
        "http://gitlab.example/api/v4",
        "https://gitlab.example.evil/api/v4",
        "https://u:p@gitlab.example/api/v4",
        "https://gitlab.example/api/v4/../user",
    ],
)
def test_gitlab_api_blocks_unsafe_urls_and_cross_host_redirects(url):
    module = adapter()
    if "example.evil" in url:
        api = module.GitLabAPI("https://gitlab.example/api/v4", "12", "secret")
        with pytest.raises(module.FeedbackError):
            api.get(url)
    else:
        with pytest.raises(module.FeedbackError):
            module.GitLabAPI(url, "12", "secret")


def test_gitlab_rejects_source_pipeline_runtime_variable_overrides(gitlab):
    module, api = gitlab
    original = api.get

    def overridden(path):
        if path == "/projects/12/pipelines/123/variables":
            return [{"key": "SKILL_EVAL_TOOLS_REF", "value": "e" * 40}]
        return original(path)

    api.get = overridden
    with pytest.raises(module.FeedbackError):
        module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)
    assert not api.writes


def test_native_pass_requires_matching_retained_tier_evidence():
    module = adapter()
    from skillevaluator.models import ValidationResult
    from skillevaluator.reporting.json_reporter import JSONReporter

    result = ValidationResult(validator_name="FORMAT", metadata={"gating": {"tier": 1, "blocking": True}})
    native = JSONReporter().render_all([result])
    payload = manifest()
    payload["tiers"][0].update(
        status="passed",
        skills=[
            {
                "path": "skills/demo",
                "status": "passed",
                "report_json": "tier1/skill-001/skillevaluator-tier1.json",
                "report_html": "tier1/skill-001/skillevaluator-tier1.html",
            }
        ],
    )
    extra = {
        "tier1/skill-001/skillevaluator-tier1.json": native,
        "tier1/skill-001/skillevaluator-tier1.html": '<script>throw "NEVER EXECUTE"</script>',
    }
    assert module.read_results(archive(payload, extra))["tiers"][0]["status"] == "passed"
    forged = json.loads(native)
    forged["results"][0]["passed"] = False
    extra["tier1/skill-001/skillevaluator-tier1.json"] = json.dumps(forged)
    with pytest.raises(module.FeedbackError):
        module.read_results(archive(payload, extra))


def _tier3_neutral_archive(status, *, complete=True, blocking=True):
    from tests.ci.test_tier3_execution_complete import complete_report

    from skillevaluator.models import ValidationResult
    from skillevaluator.reporting.json_reporter import JSONReporter

    payload = complete_report()["tier3"]
    payload["verdict"] = payload["summary"]["verdict"] = "neutral"
    if not complete:
        payload.pop("run_id")
        payload["summary"].pop("run_id")
    result = ValidationResult(
        validator_name="AGENT_EVAL",
        passed=True,
        metadata={"gating": {"tier": 3, "blocking": blocking}, "agent_eval": payload},
    )
    native = JSONReporter(expected_skill_name="demo").render_all([result])
    document = manifest()
    document["tiers"][2].update(
        status=status,
        skills=[
            {
                "path": "skills/demo",
                "status": status,
                "report_json": "tier3/skill-001/skillevaluator-tier3.json",
                "report_html": "tier3/skill-001/skillevaluator-tier3.html",
            }
        ],
    )
    document["tiers"][2].pop("reason")
    return archive(
        document,
        {
            "tier3/skill-001/skillevaluator-tier3.json": native,
            "tier3/skill-001/skillevaluator-tier3.html": "<p>Tier 3</p>",
        },
    )


@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.parametrize(("complete", "status"), [(True, "neutral"), (True, "incomplete"), (False, "incomplete")])
def test_tier3_neutral_evidence_accepts_recorded_neutral_or_producer_incomplete_downgrade(blocking, complete, status):
    module = adapter()
    retained = module.read_results(_tier3_neutral_archive(status, complete=complete, blocking=blocking))
    assert retained["tiers"][2]["skills"][0]["status"] == status


@pytest.mark.parametrize(("complete", "status"), [(True, "passed"), (False, "neutral"), (False, "passed")])
def test_tier3_neutral_evidence_is_never_promoted(complete, status):
    module = adapter()
    with pytest.raises(module.FeedbackError):
        module.read_results(_tier3_neutral_archive(status, complete=complete))


def _aggregate_layout(skills, *, directories):
    """Member names of a full three-tier aggregate as its producers retain them."""
    prefix = "skill-evaluator-results/" if directories else ""
    names = [
        "ci-results.json",
        "index.html",
        "summary.md",
        "tier1/summary.json",
        "tier1/index.html",
        "tier2/ci-tier.json",
        "tier3/ci-tier.json",
    ]
    folders = ["", "tier1/", "tier2/", "tier3/"]
    for number in range(1, skills + 1):
        leaf = f"skill-{number:03d}"
        tier1 = ("json", "html", "md", "sarif.json")
        names += [f"tier1/{leaf}/skillevaluator-tier1.{ext}" for ext in tier1]
        names += [f"tier1/{leaf}/stdout.log", f"tier1/{leaf}/stderr.log"]
        for tier in (2, 3):
            names += [f"tier{tier}/{leaf}/skillevaluator-tier{tier}.{ext}" for ext in ("json", "html", "md")]
        folders += [f"tier{tier}/{leaf}/" for tier in (1, 2, 3)]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        for folder in folders if directories else []:
            zipped.writestr(prefix + folder, "")
        for name in names:
            zipped.writestr(prefix + name, json.dumps(manifest()) if name == "ci-results.json" else "{}")
    return output.getvalue(), prefix, len(zipped.infolist())


@pytest.mark.parametrize("directories", [False, True])
def test_member_limit_admits_the_documented_maximum_skill_count_on_every_platform(directories):
    from ci import run_tier1, select_changed_skills

    module = adapter()
    assert module.MAX_SKILLS == run_tier1.MAX_SKILLS == select_changed_skills.MAX_SELECTED_SKILLS
    raw, prefix, members = _aggregate_layout(module.MAX_SKILLS, directories=directories)
    assert members <= module.MAX_FILES
    assert module.read_results(raw, prefix=prefix)["tiers"][0]["status"] == "skipped"


def test_member_limit_still_rejects_one_member_above_the_derived_bound():
    module = adapter()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        zipped.writestr("ci-results.json", json.dumps(manifest()))
        for number in range(module.MAX_FILES - 1):
            zipped.writestr(f"tier1/extra-{number:05d}.json", "{}")
    assert module.read_results(output.getvalue())["tiers"][0]["status"] == "skipped"
    with zipfile.ZipFile(output, "a") as zipped:
        zipped.writestr("tier1/one-too-many.json", "{}")
    with pytest.raises(module.FeedbackError, match="member or expansion limits"):
        module.read_results(output.getvalue())


def test_gitlab_signed_job_claims_are_not_replaced_by_ci_environment(monkeypatch):
    module = adapter()
    import time

    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = {**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key())), "kid": "gitlab", "use": "sig"}
    monkeypatch.setattr(module, "_gitlab_signing_keys", lambda _origin: [jwk])
    monkeypatch.setenv("CI_PIPELINE_ID", "999")
    now = int(time.time())
    claims = {
        "iss": "https://gitlab.example",
        "aud": module.GL_AUDIENCE,
        "iat": now,
        "nbf": now - 5,
        "exp": now + 300,
        "project_id": "12",
        "pipeline_id": "456",
        "job_id": "92",
        "ref": "main",
        "ref_protected": "true",
        "ref_type": "branch",
        "sha": MERGE,
        "pipeline_source": "pipeline",
    }
    token = jwt.encode(claims, key, algorithm="RS256", headers={"kid": "gitlab"})
    assert module.verify_gitlab_job("https://gitlab.example/api/v4", token)["pipeline_id"] == "456"
    for changes in (
        {"ref_protected": "false"},
        {"aud": "evil"},
        {"project_id": 12},
        {"pipeline_source": "merge_request_event"},
        {"sha": "branch"},
        {"exp": now - 1},
        {"iss": "https://evil.example"},
    ):
        bad = jwt.encode({**claims, **changes}, key, algorithm="RS256", headers={"kid": "gitlab"})
        with pytest.raises(module.FeedbackError):
            module.verify_gitlab_job("https://gitlab.example/api/v4", bad)


@pytest.mark.parametrize(
    "include",
    [
        [{"local": "untrusted.yml"}],
        [{"remote": "https://raw.githubusercontent.com/NVIDIA/SkillEvaluator/main/ci/gitlab/all-tiers.yml"}],
        [{"remote": "https://evil.example/source.yml"}],
        [{"local": ".gitlab/skill-feedback.yml"}],
    ],
)
def test_gitlab_transitive_config_sources_cannot_be_mutable(include):
    module = adapter()
    with pytest.raises(module.FeedbackError):
        module.validate_gitlab_config({"include": include})


def test_github_update_rechecks_bot_ownership_and_requires_exact_readback(github):
    module, api = github
    module.github_feedback(api, REPO, 123, 1)
    api.comments[0]["body"] += "\nold content"
    assert module.github_feedback(api, REPO, 123, 1)["comment_action"] == "updated"
    api.comments.clear()
    original_get = api.get

    def wrong_owner(path):
        value = original_get(path)
        if path == "/repos/acme/skills/issues/comments/51":
            value["user"]["login"] = "another-bot[bot]"
        return value

    api.get = wrong_owner
    with pytest.raises(module.FeedbackError, match="readback"):
        module.github_feedback(api, REPO, 123, 1)


@pytest.mark.parametrize("tamper", ["owner", "body"])
def test_gitlab_preserves_other_bot_marker_and_requires_its_own_readback(gitlab, tamper):
    module, api = gitlab
    api.notes = [{"id": 5, "body": module.MARKER, "author": {"id": 78, "bot": True}}]
    assert module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)["comment_action"] == "created"
    assert api.notes[0]["id"] == 5 and api.notes[0]["body"] == module.MARKER
    api.notes.pop()
    original_get = api.get

    def wrong_readback(path):
        value = original_get(path)
        if path == "/projects/12/merge_requests/7/notes/51":
            if tamper == "owner":
                value["author"]["id"] = 78
            else:
                value["body"] += "\n"
        return value

    api.get = wrong_readback
    with pytest.raises(module.FeedbackError, match="readback"):
        module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)


class _ArtifactResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.headers = {}


class _RedirectingStorage:
    """Answer the artifact endpoint with a redirect chain, then the archive bytes."""

    def __init__(self, locations, payload=b"zip-bytes"):
        self.locations, self.payload, self.requests = list(locations), payload, []

    def open(self, request, *, timeout):
        import urllib.error

        assert timeout > 0
        self.requests.append(request)
        if self.locations:
            location = self.locations.pop(0)
            raise urllib.error.HTTPError(request.full_url, 302, "Found", {"Location": location}, None)
        return _ArtifactResponse(self.payload)


def test_gitlab_follows_object_storage_artifact_redirect_without_the_token():
    module = adapter()
    api = module.GitLabAPI("https://gitlab.example/api/v4", "12", "secret")
    api.opener = _RedirectingStorage(["https://objects.example/bucket/artifact.zip?X-Amz-Signature=abc"])
    assert api.download_artifact(91) == b"zip-bytes"
    first, second = api.opener.requests
    assert first.full_url == "https://gitlab.example/api/v4/projects/12/jobs/91/artifacts"
    assert first.get_header("Private-token") == "secret"
    assert second.full_url == "https://objects.example/bucket/artifact.zip?X-Amz-Signature=abc"
    assert all("secret" not in str(value) for value in second.header_items())
    assert second.get_header("Private-token") is None and second.get_header("Authorization") is None


def test_gitlab_artifact_redirect_resolves_relative_locations_and_stays_bounded():
    module = adapter()
    api = module.GitLabAPI("https://gitlab.example/api/v4", "12", "secret")
    api.opener = _RedirectingStorage(["/-/object-storage/artifact.zip", "https://cdn.example/signed"])
    assert api.download_artifact(91) == b"zip-bytes"
    assert [r.full_url for r in api.opener.requests[1:]] == [
        "https://gitlab.example/-/object-storage/artifact.zip",
        "https://cdn.example/signed",
    ]
    api.opener = _RedirectingStorage(["https://objects.example/hop"] * module.MAX_REDIRECTS)
    assert api.download_artifact(91) == b"zip-bytes"
    api.opener = _RedirectingStorage(["https://objects.example/zip"], payload=b"x" * (module.MAX_BYTES + 1))
    with pytest.raises(module.FeedbackError, match="size limit"):
        api.download_artifact(91)


@pytest.mark.parametrize(
    "locations",
    [
        ["http://objects.example/artifact.zip"],
        ["https://user:pass@objects.example/artifact.zip"],
        ["https://user@objects.example/artifact.zip"],
        ["https://objects.example/artifact.zip\r\nX-Injected: 1"],
        ["ftp://objects.example/artifact.zip"],
        [""],
        ["https://objects.example/1", "http://objects.example/2"],
        ["https://objects.example/hop"] * 6,  # one more than MAX_REDIRECTS
    ],
)
def test_gitlab_artifact_redirect_rejects_insecure_credentialed_or_looping_targets(locations):
    module = adapter()
    api = module.GitLabAPI("https://gitlab.example/api/v4", "12", "secret")
    api.opener = _RedirectingStorage(locations)
    with pytest.raises(module.FeedbackError, match="redirect"):
        api.download_artifact(91)
    assert all(r.get_header("Private-token") is None for r in api.opener.requests[1:])
    assert all(r.full_url.startswith("https://") and "@" not in r.full_url for r in api.opener.requests)


def test_gitlab_json_api_calls_still_never_follow_redirects():
    import urllib.error

    module = adapter()
    api = module.GitLabAPI("https://gitlab.example/api/v4", "12", "secret")
    api.opener = _RedirectingStorage(["https://evil.example/data"])
    with pytest.raises(urllib.error.HTTPError):
        api.get("/projects/12/pipelines/123")
    assert len(api.opener.requests) == 1


def test_gitlab_cli_does_not_fall_back_to_job_token(monkeypatch, capsys):
    module = adapter()
    monkeypatch.delenv("SKILLEVALUATOR_GITLAB_TOKEN", raising=False)
    monkeypatch.setenv("CI_JOB_TOKEN", "never-use-me-for-notes")
    assert (
        module.main(
            [
                "--platform",
                "gitlab",
                "--api-url",
                "https://gitlab.example/api/v4",
                "--project-id",
                "12",
                "--pipeline-id",
                "123",
            ]
        )
        == 1
    )
    assert "never-use-me" not in capsys.readouterr().err


def _failed_gitlab_main(module):
    return module.main(
        [
            "--platform",
            "gitlab",
            "--api-url",
            "https://gitlab.example/api/v4",
            "--project-id",
            "12",
            "--pipeline-id",
            "123",
        ]
    )


def _safe_diagnostic(stderr, category):
    match = re.fullmatch(
        r"Trusted feedback could not validate its producer, current head, or bot-owned comment\. "
        r"\[check=feedback\.py:([1-9][0-9]*); category=" + category + r"\]\n",
        stderr,
    )
    assert match is not None
    return int(match[1])


@pytest.mark.parametrize(
    "failure,category", [("feedback", "validation"), ("http", "http_401"), ("named", "validation")]
)
def test_cli_failure_diagnostic_uses_only_trusted_line_and_finite_category(
    gitlab, monkeypatch, capsys, failure, category
):
    import urllib.error

    module, api = gitlab
    secret = "synthetic-private-token-with-claims-and-headers"
    monkeypatch.setenv("SKILLEVALUATOR_GITLAB_TOKEN", secret)
    monkeypatch.setattr(module, "GitLabAPI", lambda *_args: api)
    error = module.FeedbackError(secret)
    if failure == "http":
        error = urllib.error.HTTPError(
            "https://private.example/" + secret, 401, secret, {"Authorization": secret}, None
        )
    elif failure == "named":
        error = type(secret, (module.FeedbackError,), {})(secret)
    error.add_note("private notes: " + secret)
    original = api.get

    def get(path):
        if "/notes?" in path:
            raise error
        return original(path)

    api.get = get
    assert _failed_gitlab_main(module) == 1
    output = capsys.readouterr()
    assert output.out == ""
    line = _safe_diagnostic(output.err, category)
    assert "api.get(" in Path(module.__file__).read_text().splitlines()[line - 1]
    assert secret not in output.err and "https://" not in output.err


def test_cli_validation_diagnostic_identifies_guard_callsite_not_shared_require(gitlab, monkeypatch, capsys):
    module, api = gitlab
    monkeypatch.setenv("SKILLEVALUATOR_GITLAB_TOKEN", "synthetic-private-token")
    monkeypatch.setattr(module, "GitLabAPI", lambda *_args: api)
    api.variables = []
    original = module._require
    failed_lines = []

    def require(condition, message):
        if not condition:
            failed_lines.append(sys._getframe(1).f_lineno)
        original(condition, message)

    monkeypatch.setattr(module, "_require", require)
    assert _failed_gitlab_main(module) == 1
    assert _safe_diagnostic(capsys.readouterr().err, "validation") == failed_lines[0]


@pytest.mark.parametrize(
    "exception_name,category",
    [
        ("ExpiredSignatureError", "auth_expired"),
        ("InvalidAudienceError", "auth_audience"),
        ("InvalidSignatureError", "auth_signature"),
        ("InvalidIssuerError", "auth_issuer"),
        ("ImmatureSignatureError", "auth_immature"),
        ("MissingRequiredClaimError", "auth_missing_claim"),
        ("DecodeError", "auth_invalid"),
        ("spoofed", "validation"),
    ],
)
def test_cli_auth_diagnostic_whitelists_classes_without_messages(monkeypatch, capsys, exception_name, category):
    import jwt

    module = adapter()
    secret = "synthetic-jwt-claim-and-signature-value"
    monkeypatch.setenv("SKILLEVALUATOR_GITLAB_TOKEN", secret)
    cause = (
        type("ExpiredSignatureError", (Exception,), {})(secret)
        if exception_name == "spoofed"
        else getattr(jwt, exception_name)(secret)
    )
    cause.add_note(secret)

    def fail(*_args):
        raise module.FeedbackError(secret) from cause

    monkeypatch.setattr(module, "gitlab_feedback", fail)
    assert _failed_gitlab_main(module) == 1
    output = capsys.readouterr()
    _safe_diagnostic(output.err, category)
    assert secret not in output.err and output.out == ""


def test_cli_auth_diagnostic_reports_only_finite_wrapped_http_code(monkeypatch, capsys):
    import urllib.error

    import jwt

    module = adapter()
    secret = "synthetic-jwks-url-token-and-header"
    monkeypatch.setenv("SKILLEVALUATOR_GITLAB_TOKEN", secret)
    cause = jwt.PyJWKClientError(secret)
    cause.__cause__ = urllib.error.HTTPError(
        "https://private.example/" + secret, 404, secret, {"Authorization": secret}, None
    )
    cause.add_note(secret)

    def fail(*_args):
        raise module.FeedbackError(secret) from cause

    monkeypatch.setattr(module, "gitlab_feedback", fail)
    assert _failed_gitlab_main(module) == 1
    output = capsys.readouterr()
    _safe_diagnostic(output.err, "http_404")
    assert output.out == "" and secret not in output.err


def test_all_tier_receipt_caller_path_is_explicit_and_legacy_default_is_unchanged(monkeypatch):
    module = adapter()
    from datetime import datetime

    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    from ci import verify_provenance

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = {
        **json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key())),
        "kid": "gh",
        "alg": "RS256",
        "use": "sig",
    }
    monkeypatch.setattr(verify_provenance, "fetch_signing_keys", lambda: {"keys": [jwk]})
    run = GitHub().run
    now = int(datetime.fromisoformat(run["run_started_at"]).timestamp())
    claims = {
        "iss": verify_provenance.ISSUER,
        "aud": verify_provenance.AUDIENCE,
        "iat": now + 30,
        "nbf": now,
        "exp": now + 300,
        "repository": REPO,
        "repository_id": "12",
        "run_id": "123",
        "run_attempt": "1",
        "event_name": "pull_request",
        "workflow_ref": REPO + "/" + module.CALLER_PATH + "@refs/pull/7/merge",
        "workflow_sha": MERGE,
        "sha": MERGE,
        "ref": "refs/pull/7/merge",
        "job_workflow_ref": WORKFLOW,
        "job_workflow_sha": "c" * 40,
    }
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as zipped:
        zipped.writestr("execution.jwt", jwt.encode(claims, key, algorithm="RS256", headers={"kid": "gh"}))
    arguments = {
        "repository": REPO,
        "run": run,
        "evaluator_workflow": WORKFLOW.rsplit("@", 1)[0],
        "evaluator_sha": "c" * 40,
    }
    assert (
        verify_provenance.verify_receipt(raw.getvalue(), **arguments, caller_workflow_path=module.CALLER_PATH)[
            "workflow_sha"
        ]
        == MERGE
    )
    with pytest.raises(verify_provenance.ProvenanceError):
        verify_provenance.verify_receipt(raw.getvalue(), **arguments)


def test_gitlab_api_supports_an_https_instance_mounted_under_a_path():
    module = adapter()
    api = module.GitLabAPI("https://gitlab.example/company/api/v4", "12", "secret")
    assert api.origin == "https://gitlab.example/company"


@pytest.mark.parametrize("launcher", ["script", "runpy"])
def test_isolated_feedback_launcher_uses_trusted_siblings_from_any_consumer_directory(tmp_path, launcher):
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "ci" / "feedback.py"
    for name in ("github_api", "publish_reports", "tier_manifest", "verify_provenance"):
        (tmp_path / f"{name}.py").write_text('raise RuntimeError("CONSUMER CODE MUST NOT RUN")\n')
    for name in ("ci", "skillevaluator"):
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text('raise RuntimeError("CONSUMER CODE MUST NOT RUN")\n')
    command = [sys.executable, "-I", str(script), "--help"]
    if launcher == "runpy":
        command = [
            sys.executable,
            "-I",
            "-c",
            "import runpy,sys;sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')",
            str(script),
        ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--platform" in result.stdout
    assert "CONSUMER CODE" not in result.stdout + result.stderr
