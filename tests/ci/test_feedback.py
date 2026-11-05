# SPDX-License-Identifier: Apache-2.0
"""Trusted feedback accepts API-bound current evidence and owns only its bot notes."""

from __future__ import annotations

import base64
import copy
import importlib
import io
import json
import stat
import zipfile

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
        self.flip_head_on_recheck = False

    def get(self, path):
        self.gets.append(path)
        prefix = "/repos/" + REPO
        if path == prefix + "/actions/runs/123":
            return copy.deepcopy(self.run)
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
    assert "Tier 2 | skipped | Required" in body
    assert "Tier 3 | skipped | Advisory" in body
    assert "(tier" not in body  # archive members are not live web pages
    assert module.github_feedback(api, REPO, 123, 1)["comment_action"] == "unchanged"
    assert len(api.writes) == 1


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
    api.comments = [
        {
            "id": 5,
            "body": module.MARKER + "\n<!-- skillevaluator-all-tiers-run:124:1 -->",
            "user": {"login": "github-actions[bot]", "type": "Bot"},
        }
    ]
    assert module.github_feedback(api, REPO, 123, 1)["comment_action"] == "skipped"
    assert not api.writes


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
        self.bridge = {
            "name": "skill-evaluation-request-feedback",
            "downstream_pipeline": {"id": 456, "project_id": 12},
        }
        self.notes = []
        self.writes = []
        self.payload = archive(manifest("gitlab"), prefix="skill-evaluator-results/")
        self.user = {"id": 77, "username": "project_12_bot_abc", "bot": True}
        self.source_blob = BLOB
        self.flip_head_on_recheck = False
        self.head_checks = 0
        self.variables = [
            {"key": "SKILL_EVAL_FEEDBACK", "value": "true"},
            {"key": "SKILL_EVAL_SOURCE_PIPELINE_ID", "value": "123"},
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
        if path.startswith(root + "/pipelines/123/bridges?"):
            return [copy.deepcopy(self.bridge)]
        if path.startswith(root + "/pipelines/123/jobs?"):
            return [copy.deepcopy(self.job)]
        if path == root + "/repository/files/.gitlab-ci.yml?ref=" + SHA:
            return {"file_path": ".gitlab-ci.yml", "blob_id": self.source_blob, "commit_id": SHA, **config_content()}
        if path == root + "/repository/files/.gitlab-ci.yml?ref=" + MERGE:
            return {"file_path": ".gitlab-ci.yml", "blob_id": BLOB, "commit_id": MERGE, **config_content()}
        if path == root + "/pipelines/123/merge_requests":
            return [copy.deepcopy(self.mr)]
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


def test_gitlab_binds_protected_downstream_bridge_config_head_job_and_bot(gitlab):
    module, api = gitlab
    result = module.gitlab_feedback(api, "https://gitlab.example/api/v4", "acme/skills", 123)
    assert result["comment_action"] == "created"
    body = api.notes[0]["body"]
    assert "https://gitlab.example/acme/skills/-/jobs/91/artifacts/download" in body
    assert "https://gitlab.example/acme/skills/-/pipelines/123" in body
    assert module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)["comment_action"] == "unchanged"
    assert len(api.writes) == 1


@pytest.mark.parametrize(
    "change",
    [
        "bridge",
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
    if change == "bridge":
        api.bridge["downstream_pipeline"]["id"] = 457
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


def test_gitlab_preserves_other_bot_marker_and_requires_its_own_readback(gitlab):
    module, api = gitlab
    api.notes = [{"id": 5, "body": module.MARKER, "author": {"id": 78, "bot": True}}]
    assert module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)["comment_action"] == "created"
    assert api.notes[0]["id"] == 5 and api.notes[0]["body"] == module.MARKER
    api.notes.pop()
    original_get = api.get

    def wrong_owner(path):
        value = original_get(path)
        if path == "/projects/12/merge_requests/7/notes/51":
            value["author"]["id"] = 78
        return value

    api.get = wrong_owner
    with pytest.raises(module.FeedbackError, match="readback"):
        module.gitlab_feedback(api, "https://gitlab.example/api/v4", "12", 123)


def test_gitlab_never_forwards_token_on_artifact_redirect():
    module = adapter()
    import urllib.error

    class Redirect:
        def __init__(self):
            self.requests = []

        def open(self, request, *, timeout):
            assert timeout > 0
            self.requests.append(request)
            raise urllib.error.HTTPError(
                request.full_url, 302, "redirect", {"Location": "https://evil.example/data"}, None
            )

    api = module.GitLabAPI("https://gitlab.example/api/v4", "12", "secret")
    api.opener = Redirect()
    with pytest.raises(urllib.error.HTTPError):
        api.download_artifact(91)
    assert len(api.opener.requests) == 1
    assert api.opener.requests[0].full_url == "https://gitlab.example/api/v4/projects/12/jobs/91/artifacts"


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
