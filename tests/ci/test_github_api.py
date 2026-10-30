# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One GitHub client and one JSON policy for the trusted publisher and PR feedback."""

import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"ci/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = load("publish_reports")
feedback = load("comment_report")
render = load("render_templates")
TOOLS = [(publisher.GitHubAPI, publisher.PublicationError), (feedback.GitHubAPI, feedback.FeedbackError)]


class Response(io.BytesIO):
    def __init__(self, body: bytes, headers=None):
        super().__init__(body)
        self.headers = headers or {}


class Opener:
    def __init__(self, body=b"{}", headers=None):
        self.body, self.response_headers = body, headers
        self.requests = []

    def open(self, request, timeout):
        assert timeout == 60
        self.requests.append(request)
        return Response(self.body, self.response_headers)


def client(cls, **opener):
    api = cls("owner/consumer", "private-token")
    api.opener = Opener(**opener)
    return api


def test_both_tools_share_one_client_and_keep_their_own_diagnostics():
    for cls, error in TOOLS:
        assert cls.__mro__[1] is publisher.github_api.GitHubAPI
        assert cls.error is error
    assert publisher.github_api.GitHubAPI is feedback.github_api.GitHubAPI
    for module in (publisher, feedback):
        assert not hasattr(module, "_NoRedirect")


@pytest.mark.parametrize(("cls", "error"), TOOLS)
@pytest.mark.parametrize(
    "path",
    [
        "/repos/owner/consumer",
        "/repos/owner/consumer/pulls/1",
        f"/repos/owner/consumer/compare/{'a' * 40}...{'b' * 40}",
    ],
)
def test_this_repository_endpoints_are_allowed(cls, error, path):
    api = client(cls)
    assert api.get(path) == {}
    assert api.opener.requests[0].full_url == "https://api.github.com" + path


@pytest.mark.parametrize(("cls", "error"), TOOLS)
@pytest.mark.parametrize(
    "path",
    [
        "/repos/owner/consumer-fork/pulls/1",
        "/repos/owner/consumerx",
        "/repos/other/consumer/pulls/1",
        "/user",
        "/repos/owner/consumer/../other/pulls/1",
        "/repos/owner/consumer/pulls\\1",
        f"/repos/owner/consumer/compare/{'a' * 40}...{'b' * 40}/../other",
        f"/repos/owner/consumer/compare/{'a' * 40}..{'b' * 40}",
        f"/repos/owner/consumer-fork/compare/{'a' * 40}...{'b' * 40}",
    ],
)
def test_other_repositories_and_traversal_are_rejected_before_any_request(cls, error, path):
    api = client(cls)
    with pytest.raises(error, match="this repository"):
        api.get(path)
    assert api.opener.requests == []


@pytest.mark.parametrize(("cls", "error"), TOOLS)
@pytest.mark.parametrize("body", [b'{"id": 1, "id": 2}', b'{"value": NaN}', b"[Infinity]", b"\xff", b"{"])
def test_api_json_rejects_duplicates_non_finite_numbers_and_invalid_documents(cls, error, body):
    with pytest.raises(error, match="invalid"):
        client(cls, body=body).get("/repos/owner/consumer/pulls/1")


@pytest.mark.parametrize(("cls", "error"), TOOLS)
def test_declared_oversized_api_response_is_rejected_before_reading(cls, error, monkeypatch):
    monkeypatch.setattr(publisher.github_api, "MAX_API_BYTES", 8)
    with pytest.raises(error, match="size limit"):
        client(cls, body=b"{}", headers={"Content-Length": "9"}).get("/repos/owner/consumer")
    with pytest.raises(error, match="size limit"):
        client(cls, body=b'{"a": 1234}').get("/repos/owner/consumer")
    assert client(cls, body=b'{"a": 1}').get("/repos/owner/consumer") == {"a": 1}


@pytest.mark.parametrize(("cls", "error"), TOOLS)
def test_oversized_api_response_names_only_the_api_response(cls, error):
    with pytest.raises(error) as raised:
        client(cls, body=b"{}", headers={"Content-Length": str(16 * 1024 * 1024 + 1)}).get("/repos/owner/consumer")
    # The comment helper never downloads artifacts, so its diagnostic must not mention one.
    assert str(raised.value) == "The GitHub API response exceeds its size limit."


@pytest.mark.parametrize("headers", [{"Content-Length": "9"}, {}])
def test_oversized_artifact_download_names_the_artifact(headers, monkeypatch):
    monkeypatch.setattr(publisher, "MAX_TOTAL_BYTES", 8)
    with pytest.raises(publisher.PublicationError) as raised:
        client(publisher.GitHubAPI, body=b"123456789", headers=headers).download_artifact(456)
    assert str(raised.value) == "The downloaded artifact exceeds its size limit."


def test_publisher_api_responses_use_the_smaller_shared_cap():
    assert publisher.github_api.MAX_API_BYTES == 16 * 1024 * 1024 < publisher.MAX_FILE_BYTES


def test_write_requests_send_strict_json_with_a_tool_user_agent():
    api = client(feedback.GitHubAPI, body=b'{"id": 7}')
    assert api.post("/repos/owner/consumer/issues/1/comments", {"body": "text"}) == {"id": 7}
    assert api.patch("/repos/owner/consumer/issues/comments/7", {"body": "new"}) == {"id": 7}
    post, patch = api.opener.requests
    assert (post.get_method(), patch.get_method()) == ("POST", "PATCH")
    assert json.loads(post.data) == {"body": "text"}
    assert post.get_header("Content-type") == "application/json"
    assert post.get_header("Authorization") == "Bearer private-token"
    assert post.get_header("User-agent") == "SkillEvaluator-PR-feedback"
    with pytest.raises(ValueError):
        api.post("/repos/owner/consumer/issues/1/comments", {"value": float("nan")})
    get = client(publisher.GitHubAPI)
    get.get("/repos/owner/consumer")
    assert get.opener.requests[0].get_method() == "GET"
    assert get.opener.requests[0].data is None
    assert get.opener.requests[0].get_header("User-agent") == "SkillEvaluator-report-publisher"


def test_publication_metadata_uses_the_same_strict_json_policy(tmp_path):
    (tmp_path / "publication.json").write_text('{"pr_number": "1", "pr_number": "2"}', encoding="utf-8")
    (tmp_path / "pr-comment.md").write_text("ok", encoding="utf-8")
    with pytest.raises(feedback.FeedbackError, match="invalid JSON"):
        feedback.comment_report(object(), "owner/consumer", 123, tmp_path, "github-actions[bot]")
    (tmp_path / "publication.json").write_text('{"run_url": NaN}', encoding="utf-8")
    with pytest.raises(feedback.FeedbackError, match="invalid JSON"):
        feedback.comment_report(object(), "owner/consumer", 123, tmp_path, "github-actions[bot]")


def test_release_rendering_uses_the_shared_identity_patterns(tmp_path):
    shared = publisher.github_api
    assert publisher.SHA_PATTERN is feedback.SHA_PATTERN is shared.COMMIT_SHA
    assert publisher.REPO_PATTERN is feedback.REPOSITORY_PATTERN is shared.REPOSITORY
    assert render.github_api.GHCR_DIGEST is shared.GHCR_DIGEST
    image = "ghcr.io/nvidia/scan@sha256:" + "a" * 64
    with pytest.raises(ValueError, match="commit SHA"):
        render.render_templates(tmp_path / "out", "B" * 40, image)
    with pytest.raises(ValueError, match="GHCR"):
        render.render_templates(tmp_path / "out", "b" * 40, image.upper())


@pytest.mark.parametrize(
    ("workflow", "step", "patterns"),
    [
        (".github/workflows/publish-skill-reports.yml", "Validate trusted tool identity", ("COMMIT_SHA", "REPOSITORY")),
        (".github/workflows/evaluate-skills.yml", "Validate the immutable image reference", ("GHCR_DIGEST",)),
    ],
)
def test_workflow_copies_of_identity_patterns_match_the_shared_module(workflow, step, patterns):
    # These checks run before the tools exist on the runner, so they keep inline copies.
    import yaml

    jobs = yaml.load((ROOT / workflow).read_text(), Loader=yaml.BaseLoader)["jobs"]
    script = next(item["run"] for job in jobs.values() for item in job["steps"] if item.get("name") == step)
    for name in patterns:
        assert f'r"{getattr(publisher.github_api, name).pattern}"' in script
