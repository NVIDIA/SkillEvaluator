# SPDX-License-Identifier: Apache-2.0
"""Request a same-project publisher using only the running job's token."""

import http.client
import importlib.util
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def helper():
    path = ROOT / "ci/request_gitlab_feedback.py"
    assert path.is_file(), "The pinned feedback requester must exist"
    spec = importlib.util.spec_from_file_location("request_gitlab_feedback", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def request_env(monkeypatch):
    values = {
        "CI_API_V4_URL": "https://gitlab.example.test/gitlab/api/v4",
        "CI_SERVER_URL": "https://gitlab.example.test/gitlab",
        "CI_PROJECT_ID": "42",
        "CI_PIPELINE_ID": "123",
        "CI_DEFAULT_BRANCH": "main",
        "CI_PIPELINE_SOURCE": "merge_request_event",
        "CI_MERGE_REQUEST_SOURCE_PROJECT_ID": "42",
        "CI_JOB_TOKEN": "job-token-must-not-appear-in-output",
        "SKILLEVALUATOR_GITLAB_TOKEN": "publisher-token-must-never-be-used",
        "SKILLEVALUATOR_GITLAB_READ_TOKEN": "reader-token-must-never-be-used",
        "SKILL_EVAL_TOOLS_REF": "untrusted-extra-must-not-be-forwarded",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


def install_response(module, monkeypatch, payload=None):
    requests = []
    payload = {"id": 456, "project_id": 42, "ref": "main", "source": "pipeline"} if payload is None else payload

    class Opener:
        def open(self, request, **kwargs):
            requests.append((request, kwargs))
            return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *_handlers: Opener())
    return requests


def test_same_project_request_uses_token_body_and_only_two_selector_variables(request_env, monkeypatch, capsys):
    module = helper()
    requests = install_response(module, monkeypatch)
    assert module.main() == 0
    assert len(requests) == 1
    request, kwargs = requests[0]
    assert request.full_url == "https://gitlab.example.test/gitlab/api/v4/projects/42/trigger/pipeline"
    assert request.method == "POST"
    assert urllib.parse.parse_qs(request.data.decode()) == {
        "token": [request_env["CI_JOB_TOKEN"]],
        "ref": ["main"],
        "variables[SKILL_EVAL_FEEDBACK]": ["true"],
        "variables[SKILL_EVAL_SOURCE_PIPELINE_ID]": ["123"],
    }
    assert "token" not in request.full_url
    assert not any("token" in key.lower() for key, _value in request.header_items())
    assert 0 < kwargs["timeout"] <= 30
    output = capsys.readouterr().out
    assert "456" in output
    assert all(secret not in output for key, secret in request_env.items() if "TOKEN" in key)


@pytest.mark.parametrize(
    "key,value",
    [
        ("CI_PROJECT_ID", "42/other"),
        ("CI_PROJECT_ID", "0"),
        ("CI_PIPELINE_ID", "123&ref=evil"),
        ("CI_PIPELINE_ID", "-1"),
        ("CI_JOB_TOKEN", ""),
        ("CI_JOB_TOKEN", "token\nvalue"),
        ("CI_DEFAULT_BRANCH", ""),
        ("CI_DEFAULT_BRANCH", "../main"),
        ("CI_DEFAULT_BRANCH", "main\nother"),
        ("CI_DEFAULT_BRANCH", "refs/heads/../main"),
        ("CI_API_V4_URL", "http://gitlab.example.test/gitlab/api/v4"),
        ("CI_API_V4_URL", "https://other.example/api/v4"),
        ("CI_API_V4_URL", "https://gitlab.example.test/gitlab/api/v4?token=secret"),
        ("CI_API_V4_URL", "https://gitlab.example.test/gitlab/api/v4#fragment"),
        ("CI_API_V4_URL", "https://user:pass@gitlab.example.test/gitlab/api/v4"),
        ("CI_SERVER_URL", "https://user:pass@gitlab.example.test/gitlab"),
        ("CI_MERGE_REQUEST_SOURCE_PROJECT_ID", "43"),
        ("CI_PIPELINE_SOURCE", "web"),
    ],
)
def test_malformed_or_foreign_request_is_rejected_before_network(request_env, monkeypatch, key, value, capsys):
    module = helper()
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        module.urllib.request, "build_opener", lambda *_args: pytest.fail("Invalid input must not send credentials")
    )
    assert module.main() == 2
    output = capsys.readouterr().out
    assert all(secret not in output for name, secret in request_env.items() if "TOKEN" in name)


def test_redirect_is_refused_without_logging_token_or_response(request_env, monkeypatch, capsys):
    module = helper()
    calls = []
    real_builder = urllib.request.build_opener

    class RedirectingHTTPS(urllib.request.HTTPSHandler):
        def https_open(self, request):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, 302, request_env["CI_JOB_TOKEN"], {}, None)

    def build(*handlers):
        assert any(
            isinstance(handler, urllib.request.HTTPRedirectHandler)
            and handler.redirect_request(None, None, 302, "", {}, "https://other.example") is None
            for handler in handlers
        )
        return real_builder(*handlers, RedirectingHTTPS())

    monkeypatch.setattr(module.urllib.request, "build_opener", build)
    assert module.main() == 2
    assert len(calls) == 1
    output = capsys.readouterr().out
    assert "HTTP 302" in output and request_env["CI_JOB_TOKEN"] not in output


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 500])
def test_http_failure_logs_only_status(request_env, monkeypatch, capsys, code):
    module = helper()

    class Opener:
        def open(self, request, **_kwargs):
            raise urllib.error.HTTPError(request.full_url, code, request_env["CI_JOB_TOKEN"], {}, None)

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *_handlers: Opener())
    assert module.main() == 2
    output = capsys.readouterr().out
    assert f"HTTP {code}" in output and request_env["CI_JOB_TOKEN"] not in output
    # A refused request (e.g. no default-branch permission) skips comments, not the gate.
    assert "feedback is skipped" in output and "still gates Tier 1" in output


@pytest.mark.parametrize("error_type", [urllib.error.URLError, http.client.BadStatusLine])
def test_transport_failure_does_not_print_server_data(request_env, monkeypatch, capsys, error_type):
    module = helper()

    class Opener:
        def open(self, *_args, **_kwargs):
            raise error_type(request_env["CI_JOB_TOKEN"])

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *_handlers: Opener())
    assert module.main() == 2
    assert request_env["CI_JOB_TOKEN"] not in capsys.readouterr().out


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"id": True},
        {"id": 0},
        {"id": 456, "project_id": 43, "ref": "main", "source": "pipeline"},
        {"id": 456, "project_id": 42, "ref": "other", "source": "pipeline"},
        {"id": 456, "project_id": 42, "ref": "main", "source": "trigger"},
    ],
)
def test_response_must_identify_same_project_default_branch_pipeline(request_env, monkeypatch, payload):
    module = helper()
    install_response(module, monkeypatch, payload)
    assert module.main() == 2


def test_response_read_is_bounded(request_env, monkeypatch):
    module = helper()
    reads = []

    class Response(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return b" " * size

    class Opener:
        def open(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *_handlers: Opener())
    assert module.main() == 2
    assert reads == [1024 * 1024 + 1]
