# SPDX-License-Identifier: Apache-2.0
"""Current-head admission must use a separate read credential without redirects."""

import importlib.util
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def gate():
    spec = importlib.util.spec_from_file_location("eligible_gate", ROOT / "ci/eligible_gitlab.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def mr_env(monkeypatch):
    monkeypatch.setattr(urllib.request, "_opener", None)
    for key, value in {
        "CI_PIPELINE_SOURCE": "merge_request_event",
        "CI_API_V4_URL": "https://gitlab.example.test/api/v4",
        "CI_PROJECT_ID": "42",
        "CI_MERGE_REQUEST_IID": "1",
        "CI_MERGE_REQUEST_SOURCE_PROJECT_ID": "42",
        "CI_COMMIT_SHA": "a" * 40,
        "CI_JOB_TOKEN": "job-test-token",
        "SKILLEVALUATOR_GITLAB_TOKEN": "feedback-must-never-be-used",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SKILLEVALUATOR_GITLAB_READ_TOKEN", raising=False)


@pytest.mark.parametrize("read_token", [None, "project-read-only-test-token"])
def test_admission_uses_only_job_or_separate_read_token(mr_env, monkeypatch, read_token):
    module = gate()
    if read_token:
        monkeypatch.setenv("SKILLEVALUATOR_GITLAB_READ_TOKEN", read_token)
    requests = []

    class Opener:
        def open(self, request, *_args, **_kwargs):
            requests.append(request)
            return io.BytesIO(
                json.dumps(
                    {"state": "opened", "source_project_id": 42, "target_project_id": 42, "sha": "a" * 40}
                ).encode()
            )

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *_handlers: Opener())
    assert module.main() == 0
    headers = {k.lower(): v for k, v in requests[0].header_items()}
    assert headers == ({"private-token": read_token} if read_token else {"job-token": "job-test-token"})


def test_admission_redirect_is_not_followed_or_logged(mr_env, monkeypatch, capsys):
    module = gate()
    monkeypatch.setenv("SKILLEVALUATOR_GITLAB_READ_TOKEN", "project-read-only-test-token")
    calls = []
    real_builder = urllib.request.build_opener

    class RedirectingHTTPS(urllib.request.HTTPSHandler):
        def https_open(self, request):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, 302, "sensitive response body", {}, None)

    def build(*handlers):
        assert any(
            isinstance(h, urllib.request.HTTPRedirectHandler)
            and h.redirect_request(None, None, 302, "", {}, "https://other.example") is None
            for h in handlers
        )
        return real_builder(*handlers, RedirectingHTTPS())

    monkeypatch.setattr(module.urllib.request, "build_opener", build)
    assert module.main() == 2
    output = capsys.readouterr().out
    assert "HTTP 302" in output
    assert "project-read-only-test-token" not in output and "sensitive response body" not in output
    assert len(calls) == 1


def test_admission_auth_failure_reports_only_numeric_status(mr_env, monkeypatch, capsys):
    module = gate()

    class Opener:
        def open(self, request, *_args, **_kwargs):
            raise urllib.error.HTTPError(request.full_url, 404, "job-test-token", {}, None)

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *_handlers: Opener())
    assert module.main() == 2
    output = capsys.readouterr().out
    assert "HTTP 404" in output and "job-test-token" not in output
